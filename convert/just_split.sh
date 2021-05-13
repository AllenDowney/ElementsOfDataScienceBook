#!/bin/bash
# Build the LaTex book version

# copy the figures
rsync -a ~/ElementsOfDataScience/figs .

for NOTEBOOK in $@
do
    # convert markdown to LaTeX
    FLAGS="--listings --top-level-division=chapter"
    MDFILE=${NOTEBOOK%.ipynb}.md
    TEXFILE=${NOTEBOOK%.ipynb}.tex
    echo "pandoc $FLAGS -s $MDFILE -o $TEXFILE"
    pandoc $FLAGS -s $MDFILE -o $TEXFILE

    # remove front and backmatter from the chapters
    # (and make a few text substitutions)
    echo "python split.py $TEXFILE"
    python split.py $TEXFILE

done
