# Many-N CE2, one-excitation TC benchmark

This folder implements the many-molecule CE2 hierarchy from `Discussion.txt`.
The state includes the intermolecular two-body moments `M`, `P`, `R`, and `Z`,
so it is distinct from the single-emitter CE2 closure used for the `N=1`
CE2-CE5 benchmark.

Defaults:

- `N = 5`
- `N_exc = 1`
- initial state `|1 photon> x |g_1 ... g_N>`
- `omega_c = omega_0 = omega_d = 2.0 eV`
- no drive or dissipation
- collective coupling `g_collective = 0.1 eV`, with `g_single = 0.1/sqrt(N) eV`
- time grid `0..200 fs` with `4001` points

Run:

```bash
python CE2_many_molecules.py
python plotting.py
python sweep_correlations.py
python plot_correlation_scaling.py
```

Main outputs:

- `Photon_number_CE2.dat`
- `Excited_population_CE2.dat` for total molecular excitation
- `Excited_population_per_molecule_CE2.dat`
- `Total_excitation_CE2.dat`
- `CE2_observables.dat`
- `CE2_moments.npz`
- `Photon_number_exact.dat` and `Excited_population_exact.dat` for the
  one-excitation TC comparison reference. CE2 is an approximate closure here;
  exact files are included to show the deviation for this controlled case.
- `correlation_sweep.npz` and `correlation_scaling_summary.dat` for the
  multi-`N` correlation sweep
- `CE2_correlation_scaling.pdf/png`
