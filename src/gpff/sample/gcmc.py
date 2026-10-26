"""Grand-canonical Monte Carlo on the pseudo-PES."""

# python
import math
from typing import Dict, Optional, Sequence, Tuple

import torch

# gpff
from gpff import keys
from gpff.sample.base import Calculator
from gpff.utils.neighbors import compute_neighbors

# schnetpack
from schnetpack import properties


__all__ = ["GrandCanonicalSampler"]


class GrandCanonicalSampler:
    """
    Grand-canonical Monte Carlo on a learned pseudo-PES, so the atom count is
    sampled together with the geometry instead of being fixed before sampling
    starts.

    Needs a conservative surface (experiment=gpff_conservative): every move is
    accepted on an energy difference, which a direct force head has none of.
    That is the whole point of the construction. A denoiser gives no signal
    about whether a dimension should exist, so trans-dimensional diffusion has
    to learn an insertion distribution, which Branching Flows
    (arXiv:2511.09465) reports as the hard part; a potential gives the
    criterion for free, as the textbook Metropolis acceptance of an insertion.

    Three moves in the mu-V-T ensemble, at activity z = exp(mu / kT) with the
    thermal wavelength absorbed into it. Insertions and deletions both draw an
    element first, so the chain is the standard multicomponent one and N below
    is the count of that element, not of all atoms:

        displace   one atom by a Gaussian step, accepted on exp(-dE / kT)
        insert     one atom of the drawn element, uniformly in the box,
                   accepted on z V / (N + 1) * exp(-dE / kT)
        delete     one atom of the drawn element, uniformly among them,
                   accepted on N / (z V) * exp(-dE / kT)

    The box is fixed once from the starting geometry and never follows the
    configuration: a box that moved with the molecule would give the forward
    and reverse moves different volumes and break detailed balance. Moves that
    would leave the min_atoms/max_atoms range are rejected rather than
    clipped, which keeps the chain a valid one on the restricted state space.

    The chemical potential is the knob on size: raising it pays for more
    atoms, lowering it strips them. Both it and the temperature are in the
    units of the pseudo-energy, |x_t - x_0|^2 summed over the molecule, which
    scales as a squared length -- so they have to be set against the scale the
    model was trained on and have nothing to do with Kelvin or kcal/mol.

    One molecule at a time: an accepted insertion or deletion invalidates the
    neighbor list and every atom index, so batching would mean regrouping the
    whole batch after each move. This is not wired into ``gpff_generate``,
    whose loop assumes a fixed atom count; drive it directly instead:

        from gpff.sample import Calculator, GrandCanonicalSampler

        sampler = GrandCanonicalSampler(
            calculator=Calculator(model=model, device="cuda"),
            chemical_potential=-1.0,
            temperature=0.05,
        )
        result = sampler.run(batch)
        print(result["n_atoms"], result["energy"], result["acceptance"])
    """

    MOVES: Tuple[str, str, str] = ("displace", "insert", "delete")

    def __init__(
        self,
        calculator: Calculator,
        chemical_potential: float,
        temperature: float = 1.0,
        n_sweeps: int = 100,
        displacement: float = 0.1,
        box_margin: float = 2.0,
        atom_types: Sequence[int] = (1, 6, 7, 8, 9),
        move_probabilities: Sequence[float] = (0.5, 0.25, 0.25),
        min_atoms: int = 1,
        max_atoms: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        """
        Args:
            calculator: the Calculator running model inference; its model needs
                an energy output.
            chemical_potential: mu, in the units of the pseudo-energy. The knob
                on how many atoms the chain settles at.
            temperature: kT, in the same units.
            n_sweeps: how many sweeps to run; one sweep attempts as many moves
                as the molecule currently has atoms.
            displacement: standard deviation of a displacement move.
            box_margin: how far the fixed insertion box extends past the
                starting geometry's bounding box, on every side.
            atom_types: the atomic numbers that may be inserted or deleted.
            move_probabilities: the relative weights of displace, insert and
                delete, normalized here.
            min_atoms: moves that would go below this many atoms are rejected.
            max_atoms: moves that would go above this many atoms are rejected;
                None leaves the count unbounded above.
            seed: seed of this chain's own generator; None leaves it unseeded.
        """
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}.")
        if displacement <= 0:
            raise ValueError(f"displacement must be positive, got {displacement}.")
        if box_margin < 0:
            raise ValueError(f"box_margin must be non-negative, got {box_margin}.")
        if len(move_probabilities) != len(self.MOVES):
            raise ValueError(
                f"move_probabilities needs one weight per move {self.MOVES}, got "
                f"{tuple(move_probabilities)}."
            )
        weights = torch.as_tensor(list(move_probabilities), dtype=torch.float64)
        if (weights < 0).any() or weights.sum() <= 0:
            raise ValueError(
                f"move_probabilities must be non-negative with a positive sum, got "
                f"{tuple(move_probabilities)}."
            )
        if min_atoms < 1:
            raise ValueError(f"min_atoms must be at least 1, got {min_atoms}.")
        if max_atoms is not None and max_atoms < min_atoms:
            raise ValueError(
                f"max_atoms {max_atoms} is below min_atoms {min_atoms}."
            )
        if len(atom_types) == 0:
            raise ValueError("atom_types must not be empty.")

        self.calculator = calculator
        self.chemical_potential = chemical_potential
        self.temperature = temperature
        self.n_sweeps = n_sweeps
        self.displacement = displacement
        self.box_margin = box_margin
        self.atom_types = [int(number) for number in atom_types]
        self.move_weights = weights / weights.sum()
        self.min_atoms = min_atoms
        self.max_atoms = max_atoms
        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(seed)

    # ---------------------------------------------------------------- randomness

    def _uniform(self) -> float:
        return float(torch.rand((), generator=self.generator))

    def _randint(self, upper: int) -> int:
        return int(torch.randint(upper, (), generator=self.generator))

    def _like(self, size: int, reference: torch.Tensor, normal: bool) -> torch.Tensor:
        """
        A CPU draw moved onto the reference, so one seed reproduces the chain
        on any device.
        """
        draw = (torch.randn if normal else torch.rand)(
            size, generator=self.generator, dtype=reference.dtype
        )
        return draw.to(reference.device)

    def _pick_move(self) -> str:
        return self.MOVES[
            int(torch.multinomial(self.move_weights, 1, generator=self.generator))
        ]

    # ------------------------------------------------------------------- batching

    def _rebuild(
        self,
        template: Dict[str, torch.Tensor],
        positions: torch.Tensor,
        numbers: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        A single-molecule batch at these positions and elements, with a fresh
        neighbor list, which an insertion or deletion invalidates along with
        every atom index.
        """
        n_atoms = positions.shape[0]
        device = positions.device
        batch = {
            properties.idx: template[properties.idx][:1],
            properties.n_atoms: torch.full(
                (1,),
                n_atoms,
                dtype=template[properties.n_atoms].dtype,
                device=device,
            ),
            properties.Z: numbers,
            properties.R: positions,
            properties.cell: template[properties.cell][:1],
            properties.pbc: template[properties.pbc].view(-1, 3)[:1],
            properties.idx_m: torch.zeros(n_atoms, dtype=torch.long, device=device),
        }
        batch = compute_neighbors(batch, fully_connected=True, device=device)
        batch[keys.converged] = torch.zeros(1, dtype=torch.bool, device=device)
        batch[keys.nfe] = torch.zeros(1, dtype=torch.int32, device=device)
        return batch

    def _energy(self, state: Dict[str, torch.Tensor]) -> float:
        return float(
            self.calculator.calculate(state, energy=True)[keys.energy].reshape(())
        )

    # -------------------------------------------------------------------- moves

    def _propose(
        self,
        move: str,
        positions: torch.Tensor,
        numbers: torch.Tensor,
        low: torch.Tensor,
        high: torch.Tensor,
        log_volume: float,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, float]]:
        """
        Propose one move, as new positions, new elements and the log of the
        ensemble prefactor its acceptance carries. None means the move is not
        available and counts as a rejection.
        """
        n_atoms = positions.shape[0]
        log_activity = self.chemical_potential / self.temperature

        if move == "displace":
            if n_atoms == 0:
                return None
            new_positions = positions.clone()
            index = self._randint(n_atoms)
            new_positions[index] = new_positions[index] + self.displacement * self._like(
                3, positions, normal=True
            )
            return new_positions, numbers, 0.0

        element = self.atom_types[self._randint(len(self.atom_types))]

        if move == "insert":
            if self.max_atoms is not None and n_atoms >= self.max_atoms:
                return None
            fraction = self._like(3, positions, normal=False)
            new_atom = low + fraction * (high - low)
            new_positions = torch.cat([positions, new_atom[None]], dim=0)
            new_numbers = torch.cat(
                [
                    numbers,
                    torch.full(
                        (1,), element, dtype=numbers.dtype, device=numbers.device
                    ),
                ]
            )
            # z V / (N + 1), with N the count of this element
            n_element = int((numbers == element).sum())
            return (
                new_positions,
                new_numbers,
                log_activity + log_volume - math.log(n_element + 1),
            )

        if move == "delete":
            if n_atoms <= self.min_atoms:
                return None
            candidates = torch.where(numbers == element)[0]
            if candidates.numel() == 0:
                return None
            index = int(candidates[self._randint(candidates.numel())])
            keep = torch.ones(n_atoms, dtype=torch.bool, device=positions.device)
            keep[index] = False
            # N / (z V), with N the count of this element before the deletion
            return (
                positions[keep],
                numbers[keep],
                math.log(candidates.numel()) - log_activity - log_volume,
            )

        raise ValueError(f"Unknown move {move!r}; expected one of {self.MOVES}.")

    # ---------------------------------------------------------------------- run

    @torch.no_grad()
    def run(self, batch: Dict[str, torch.Tensor]) -> Dict[str, object]:
        """
        Run the chain on one molecule.

        Args:
            batch: a single-molecule schnetpack batch; its geometry is the
                starting configuration and fixes the insertion box.

        Returns:
            A dict holding the final ``batch``, its ``energy`` and ``n_atoms``,
            and the per-move ``acceptance`` counts.
        """
        self.calculator.check_model()
        batch = self.calculator.cast_batch(batch)
        n_molecules = int(batch[properties.n_atoms].numel())
        if n_molecules != 1:
            raise ValueError(
                f"{type(self).__name__} runs one molecule at a time, got {n_molecules}."
            )

        positions = batch[properties.R].detach().clone()
        numbers = batch[properties.Z].detach().clone()

        # fixed once: a box that followed the configuration would give the
        # forward and reverse moves different volumes
        low = positions.min(dim=0).values - self.box_margin
        high = positions.max(dim=0).values + self.box_margin
        log_volume = float((high - low).log().sum())

        state = self._rebuild(batch, positions, numbers)
        energy = self._energy(state)
        counts = {move: {"attempted": 0, "accepted": 0} for move in self.MOVES}

        for _ in range(self.n_sweeps):
            for _ in range(max(positions.shape[0], 1)):
                move = self._pick_move()
                counts[move]["attempted"] += 1
                proposal = self._propose(
                    move, positions, numbers, low, high, log_volume
                )
                if proposal is None:
                    continue

                new_positions, new_numbers, log_prefactor = proposal
                new_state = self._rebuild(batch, new_positions, new_numbers)
                new_energy = self._energy(new_state)
                log_accept = log_prefactor - (new_energy - energy) / self.temperature
                if log_accept >= 0.0 or math.log(self._uniform()) < log_accept:
                    positions, numbers = new_positions, new_numbers
                    state, energy = new_state, new_energy
                    counts[move]["accepted"] += 1

        return {
            "batch": state,
            "energy": energy,
            "n_atoms": int(numbers.numel()),
            "acceptance": counts,
        }
