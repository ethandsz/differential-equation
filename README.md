# 3D Pollutant Dispersion Simulation

This Python script simulates the dispersion of a pollutant in a 3D fluid domain using the finite volume method. It models the transient convection-diffusion equation using the FiPy library. The simulation provides visualization via either Mayavi or Matplotlib.

## Requirements

* Python 3.7+
* FiPy
* matplotlib
* numpy
* mayavi (optional, for 3D interactive visualization)

## Running the Simulation

```bash
python3 pollution_sim.py [options]
```

### Available Arguments

| Argument       | Type  | Default | Description                                                              |
| -------------- | ----- | ------- | ------------------------------------------------------------------------ |
| `--useMatplot` | bool  | `False` | Enable Matplotlib 3D plot visualization (simpler alternative to Mayavi). |
| `--useMayavi`  | bool  | `True`  | Enable Mayavi 3D viewer (interactive visualization).                     |
| `--simSteps`   | int   | `300`   | Number of simulation time steps.                                         |
| `--xFlow`      | float | `0.0`   | Flow velocity in the X direction.                                        |
| `--yFlow`      | float | `0.0`   | Flow velocity in the Y direction.                                        |
| `--zFlow`      | float | `1.0`   | Flow velocity in the Z direction.                                        |

### Example Usage

Run with Mayavi viewer (default):

```bash
python3 pollution_sim.py
```

Use Matplotlib viewer instead:

```bash
python3 pollution_sim.py --useMatplot true --useMayavi false
```

## Description

* Sets up a 3D cubic mesh using FiPy.
* Pollutant near the bottom-center of the domain.
* Solves the transient convection-diffusion PDE over a number of time steps.
* Visualizes pollutant spread:

  * Matplotlib: Static 3D scatter plot of contaminated zones.
  * Mayavi: Full 3D interactive view of the solution field.
