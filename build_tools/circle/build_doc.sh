#!/usr/bin/env bash
set -x
set -e

MAKE_TARGET=html

#source activate testenv

# Create .rst versions of the .md files we need for the docs.
pandoc --from=markdown --to=rst --output=README.rst README.md
pandoc --from=markdown --to=rst --output=RELEASE.rst RELEASE.md
pandoc --from=markdown --to=rst --output=ATTENTION.rst deep_qa/layers/attention/README.md
pandoc --from=markdown --to=rst --output=TENSOR.rst deep_qa/tensors/README.md
pandoc --from=markdown --to=rst --output=SIMILARITY.rst deep_qa/tensors/similarity_functions/README.md
pandoc --from=markdown --to=rst --output=TRAINING.rst deep_qa/training/README.md

# The pipefail is requested to propagate exit code
set -o pipefail && cd doc && make $MAKE_TARGET 2>&1 | tee ~/log.txt

echo "Finished building docs."
echo "Artifacts in $CIRCLE_ARTIFACTS"

# Cleanup .rst files we created.
rm ../README.rst
rm ../RELEASE.rst
rm ../ATTENTION.rst
rm ../TENSOR.rst
rm ../SIMILARITY.rst
rm ../TRAINING.rst