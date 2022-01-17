import sys
import re

# replace figure URLs with references to local files
# replace references to PNG and SVG with PDF
# for source listings in Python, add `style=source`
# for output listings, add `style=output`

subs = {
r'\\includegraphics(.*){https://.*/([^/]*)}': r'\\includegraphics\1{figs/\2}',
#r'_0.png': '_0.pdf',
#r'_0.svg': '_0.pdf',
#r'language=Python': r'language=Python,style=source',
#r'begin{lstlisting}$': r'begin{lstlisting}[style=output]',
r'language=Python': '',
'\ufeff': '',
r'\\tightlist': '',
r'\\toprule': r'\\midrule',
r'\\bottomrule': r'\\midrule',
}

def write_line(fout, line):
    for pattern, repl in subs.items():
        line = re.sub(pattern, repl, line)
    #if line.startswith('\\label'):
    #    return
    fout.write(line + '\n')

def write_chapter(i, t):
    fout = open(filename, 'w')
    line = t[i]
    write_line(fout, line)
    i += 1

    while i < len(t):
        line = t[i]
        if line.startswith('\\backmatter'):
            fout.close()
            return i

        write_line(fout, line)
        i += 1

    fout.close()
    return i

filename = sys.argv[1]

lines = open(filename).read()

# get the source code and output into shape
pattern = r'\\end{lstlisting}\s*\\begin{lstlisting}\n([^\\]*)\n\\end'
repl = r'(@\\dashfill@)\n@@@\1@@@\n\\end'
lines = re.sub(pattern, repl, lines)

pattern = r"""\\begin{figure}
\\centering
\\includegraphics{(.*)}
\\caption{.*}
\\end{figure}"""

repl = r"""\\begin{center}
\\includegraphics[width=4in]{\1}
\\end{center}"""

lines = re.sub(pattern, repl, lines)

t = lines.split('\n')

i = 0
while i < len(t):
    line = t[i]
    if line.startswith('\\hypertarget'):
        i = write_chapter(i, t)
    else:
        i += 1
