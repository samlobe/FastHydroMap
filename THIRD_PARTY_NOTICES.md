# Third-party notices

FastHydroMap's source code and trained weights are distributed under the
[MIT license](LICENSE). Bundled parameter data retain their upstream terms.
Dependencies are installed separately under their respective licenses.

## a99SB-disp parameter data

`src/FastHydroMap/v2/chemistry.json` contains heavy-atom parameter tables adapted
from HydroMap's OpenMM conversion of the a99SB-disp force field. Hydrogen
charges are combined with their bonded heavy atoms; this is a feature table,
not a complete force field for molecular dynamics.

The [GROMACS distribution](https://github.com/paulrobustelli/Force-Fields)
identifies [VIPARR-FF](https://github.com/DEShawResearch/viparr-ffpublic) as its
upstream parameter source. The VIPARR-FF copyright notice, redistribution
conditions, and disclaimer are retained in
[licenses/VIPARR-FF-LICENSE.txt](licenses/VIPARR-FF-LICENSE.txt) and included
in both the wheel and source distribution. `LicenseRef-VIPARR-FF` in the
package metadata refers to that license.

Reference: Robustelli, P.; Piana, S.; Shaw, D. E. Developing a molecular dynamics
force field for both folded and disordered protein states. *PNAS* **2018**,
115, E4758–E4766. [doi:10.1073/pnas.1800690115](https://doi.org/10.1073/pnas.1800690115).

No endorsement by the force-field authors or D. E. Shaw Research is implied.
