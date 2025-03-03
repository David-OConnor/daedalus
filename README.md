# Bio Chem Viewer

For viewing and performing minor edits on moleculars; espsecially proteins and nucleic acids.

Conceptually similar to [PyMol](https://www.pymol.org/), Discovery Studio, Chimera etc.

## Functionality

- Viewing proteins and other molecules' 3D structure.


## Goals
- Fast
- Easy-to-use
- Practical workflow
- Updates responsive to user feedback


## File formats
This application can load data from mmCIF and PDB files.


## The camera

The camera controls operate on the perspective of the viewer, vice the molecule. You can move and rotate the camera
in 6 degrees of freedom, allowing you to easily view the molecule from any perspective.

### Mouse controls:
- Hold the **left mouse button while dragging** to rotate the camera in pitch and yaw.
- Hold the **middle mouse button while dragging** to move the camera left, right, up, and down.
- **Scroll** to move the camera forward and backwards.
- **Scroll while holding left mouse button** to roll.
- **Right click** to select the atom or residue under the cursor.


### Camera Hotkeys (Also available as GUI buttons)
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


### Erratta
- Cartoon view (Showing helices and sheets) is currently unavailable.
- The only van der Waals surface view currently available is *dots*. No mesh yet.
- The GUI buttons for movement and rotation stop working after 1 second of movement.


