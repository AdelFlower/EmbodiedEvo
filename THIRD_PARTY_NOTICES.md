# Third-party material and attribution

## EmbodiedEvo

DynamicsAdapt is based on EmbodiedEvo and retains its Agent execution, isolation,
provider-home staging, process handling and evidence-recording infrastructure.
The current project changes the research focus to dynamics diagnosis and RL
recovery. See [source provenance](docs/SOURCE.md) and the original
[EmbodiedEvo paper](https://arxiv.org/abs/2610.18867).

The pre-refactor repository is preserved at Git commit
`37a151311bb3989596bb79a33ccc18121408b817`. `SOURCE_MANIFEST.json` is an unchanged
historical export record, not a license or current-tree inventory.

## Historical RoboDojo task recipes

The former `tasks/RoboDojo/recipe/` directory contained 42 Markdown recipes from
[RoboProbe/RoboProbe](https://github.com/RoboProbe/RoboProbe), upstream commit
`52300752080325086095854e01977a8cf281b684`, path
`policy/RoboDojo_Agent_L3_Inspect/recipes`.

Recipes are removed from the active project and are not supplied to the
DynamicsAdapt Agent. Their original texts, Apache-2.0 license and checksum
manifest remain unchanged in Git history at the pre-refactor commit:

| Historical path | Record |
|---|---|
| `tasks/RoboDojo/recipe/*.md` | Original recipe text and attribution README |
| `tasks/RoboDojo/recipe/LICENSE` | Upstream Apache-2.0 license |
| `tasks/RoboDojo/recipe/manifest.json` | Upstream revision and recipe checksums |

For example, inspect the retained license with:

```bash
git show 37a151311bb3989596bb79a33ccc18121408b817:tasks/RoboDojo/recipe/LICENSE
```

The historical recipes were supplied task knowledge, not learned Agent skills.
Their license does not assign a license to the rest of this project.

## External dependencies

RoboDojo, Isaac Sim, Isaac Lab, CuRobo, simulator assets, PyTorch, the Agent CLI
and model services retain their own terms and licenses. Their native sources,
assets, images and credentials are not bundled here. Use a separately provisioned
repository-local simulator snapshot and preserve its existing revision/patch
provenance. Do not modify vendor sources during experiments or redirect imports
to another development checkout.

A project-wide license for DynamicsAdapt has not been selected.
