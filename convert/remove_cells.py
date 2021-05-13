import sys
import nbformat as nbf

filename = sys.argv[1]
print('Clearing cells from', filename)
ntbk = nbf.read(filename, nbf.NO_CONVERT)

for cell in ntbk.cells:

    # if a cell has a remove-cell tag, remove the source and outputs
    cell_tags = cell.get('metadata', {}).get('tags', [])
    #print(cell_tags)
    if ('remove-cell' in cell_tags or
        'remove-print' in cell_tags):
        cell['source'] = ''
        cell['cell_type'] = 'raw'
        if hasattr(cell, 'outputs'):
            del cell['outputs']
        if hasattr(cell, 'execution_count'):
            del cell['execution_count']

nbf.write(ntbk, filename)
