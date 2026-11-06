//! Logica spectrală a urnelor — SCHELET (protocol S18 §3, decizia R7).
//!
//! Port STRUCTURAL al formulelor deja sigilate în `tu4/PROTOCOL_TU4_IP_H1_MA.md` și
//! `tu4/PROTOCOL_TU4_IP_H3_AD.md` (implementarea de referință: `tu4/ip_core.py`, `logw_h1`/`logw_h3`,
//! evaluate la momentul curent, N2 = N). În S18 aceste module sunt testate NUMAI pe fixturi sintetice;
//! paritatea cu `ip_core.py` și orice rulare pe arhiva reală cer protocol separat (S19+).
//! Nicio funcție de aici nu produce bilete, emisii sau decontări.
//!
//! Adaos S19 (`tu4/PROTOCOL_SUPERQUANTUM_S19.md`): `fisher` (densitățile π, Π, ρ ale legii Fisher a modelului),
//! `parity` (verificarea încrucișată cu `ip_core.py` înghețat, toleranță 1e-12) și submodulele spectrale `frft`
//! (FrFT Candan–Kutay–Ozaktas) și `admissible` (filtrul S17 pe măști `u64`). Filtrul de admisibilitate este un
//! strat de AFIȘARE pe emisiile sigilate (citite read-only), nu o emisie nouă.

pub mod fisher;
pub mod h1_mass;
pub mod h3_aero;
pub mod parity;
pub mod spectral;

pub use fisher::{density_report, inclusion, info_d, pair_inclusion, DensityReport};
pub use h1_mass::{mass_correlation, H1Mass, MassCorrelation};
pub use h3_aero::{basis_constant_residual, basis_gram_residual, cosine_basis, H3Aero, K_AD};
pub use spectral::{
    label_dft, neumaier_sum, pair_correlation, parseval_relative_residual, CountState, HermitianMatrix, SpectralError,
    UrnSpec,
};
