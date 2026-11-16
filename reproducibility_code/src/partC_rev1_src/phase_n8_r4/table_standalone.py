"""Build a portable LaTeX table from its local display CSV; no project dependency."""
from pathlib import Path
import csv

ROOT=Path(__file__).resolve().parent


def escape(value):
    replacements={'\\':r'\textbackslash{}','&':r'\&','%':r'\%','$':r'\$','#':r'\#','_':r'\_',
        '{':r'\{','}':r'\}','~':r'\textasciitilde{}','^':r'\textasciicircum{}'}
    return ''.join(replacements.get(c,c) for c in str(value))


def main():
    with (ROOT/'table.csv').open(encoding='utf-8',newline='') as f:
        reader=csv.DictReader(f);columns=reader.fieldnames;rows=list(reader)
    if not columns or not rows:raise ValueError('A table must contain real rows')
    with (ROOT/'table_metadata.csv').open(encoding='utf-8',newline='') as f:metadata=next(csv.DictReader(f))
    # Account for both sides of every column's padding, including wide tables.
    layout=''.join(r'p{\dimexpr\linewidth/'+str(len(columns))+r'-2\tabcolsep\relax}' for _ in columns)
    lines=[r'\begingroup\footnotesize',r'\setlength{\tabcolsep}{3pt}',r'\begin{longtable}{'+layout+'}',
        r'\caption{'+escape(metadata['caption'])+r'}\label{'+metadata['label']+r'}\\',r'\toprule',
        ' & '.join(r'\textbf{'+escape(c)+'}' for c in columns)+r' \\',r'\midrule',r'\endhead']
    lines.extend(' & '.join(escape(row[c]) for c in columns)+r' \\' for row in rows)
    lines.extend([r'\bottomrule',r'\end{longtable}',r'\endgroup'])
    (ROOT/'table.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Built table from',len(rows),'local CSV rows')


if __name__=='__main__':main()
