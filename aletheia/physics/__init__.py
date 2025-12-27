"""Physics simulation module for Project Aletheia"""

from aletheia.physics.mpm import MPMSimulator, TorchMPMSimulator, ParticleSystem
from aletheia.physics.fields import NeuralFieldSolver, GridField
from aletheia.physics.pde import (
    PDEPerturbation,
    NavierStokesSolver,
    ReactionDiffusionSolver,
    CoupledPDESolver,
)

__all__ = [
    "MPMSimulator",
    "TorchMPMSimulator",
    "ParticleSystem",
    "NeuralFieldSolver",
    "GridField",
    "PDEPerturbation",
    "NavierStokesSolver",
    "ReactionDiffusionSolver",
    "CoupledPDESolver",
]
