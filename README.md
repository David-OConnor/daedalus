# Daedelus molecular viewer

For viewing and performing minor edits on moleculars; espsecially proteins and nucleic acids.

Conceptually similar to [PyMol](https://www.pymol.org/), [Chimera](https://www.cgl.ucsf.edu/chimera/), and Discovery Studio.
Designed to be as easy to use as possible.

## Functionality

- View the 3D structure of proteins and small molecules
- Visualize ligand docking
- WIP: This software is a platform for molecular docking, and ab-initio simulations.


## Goals
- Fast
- Easy-to-use
- Practical workflow
- Updates responsive to user feedback


## File formats
- Proteins: mmCIF and PDB
- Small molecules: SDF, Mol2, and PDBQT


## The camera

There are two camera control schemes, selectable using buttons in the *camera* section of the GUI.

### Free camera
The free camera controls is intended to be used with a keyboard and mouse together. They operate on the perspective of 
the viewer, vice the molecule. You can move and rotate and move the camera
in 6 degrees of freedom, allowing you to easily view the molecule from any perspective.

#### Mouse controls:
- Hold the **left mouse button while dragging** to rotate the camera in pitch and yaw.
- Hold the **middle mouse button while dragging** to move the camera left, right, up, and down.
- **Scroll** to move the camera forward and backwards.
- **Scroll while holding left mouse button** to roll.
- **Right click** to select the atom or residue under the cursor.


#### Camera Hotkeys (Also available as GUI buttons)
- **W**: Move forward
- **A**: Move right
- **A**: Move left
- **D**: Move back
- **Space**: Move up
- **C**: Move down
- **Q**: Roll counter-clockwise
- **R**: Roll clockwise

- **Shift** (left): Hold to increase camera movement and rotation speed.

- **Left arrow**: Select previous residue
- **Right arrow**: Select next residue


### Arc camera
Similar to traditional molecular viewing software. The camera arcs (or orbits) around the molecule, when holding the left
mouse button and dragging. Other controls, like scroll wheel and middle mouse, operate similar to the free camera.
If *orbit sel* is set in the GUI, the orbit center will be the selected atom or residue, vice the molecule center.


### Erratta
- Cartoon view (Showing helices and sheets) is currently unavailable.
- The only van der Waals surface view currently available is *dots*, and it's slow to build. Mesh WIP


