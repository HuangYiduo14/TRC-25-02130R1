"""Synthetic typography checks; no campaign CSVs or numerical models opened."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import RendererAgg
from matplotlib.font_manager import FontProperties
import pandas as pd
import pytest

from partC_rev1_src.pacnhn_v1.publication_templates import build_table as b


@pytest.fixture
def renderer():
    return RendererAgg(2376, 1548, 180)


@pytest.mark.parametrize('weight', ['normal', 'bold'])
@pytest.mark.parametrize('text', [
    'Recorded regions / TRAIN target met',
    'Full B WMAPE (%)',
    'NEW_DETAILED_REGIONAL_FW_ON_FROZEN_P',
    'WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW',
])
def test_actual_width_and_complete_tokens(renderer, weight, text):
    font = FontProperties(family='DejaVu Sans', size=9, weight=weight)
    wrapped = b.wrapped(text, 92, renderer, font)
    assert ''.join(wrapped.split()) == ''.join(text.split())
    assert all(renderer.get_text_width_height_descent(line, font, False)[0] <= 92
               for line in wrapped.splitlines())
    assert '\n' in wrapped


def test_bold_header_is_measured_separately_and_newlines_preserved(renderer):
    normal = FontProperties(family='DejaVu Sans', size=9)
    bold = FontProperties(family='DejaVu Sans', size=9, weight='bold')
    phrase = 'WWWWW WWWWW'
    narrow = renderer.get_text_width_height_descent(phrase, normal, False)[0] + .1
    assert b.wrapped(phrase, narrow, renderer, normal) == phrase
    assert '\n' in b.wrapped(phrase, narrow, renderer, bold)
    assert b.wrapped('first\n\nlast', 300, renderer, normal) == 'first\n\nlast'


def test_post_draw_guard_rejects_actual_horizontal_and_vertical_overflow():
    for text, height in [('W'*90, .2), ('one\ntwo\nthree\nfour\nfive', .03)]:
        fig = plt.figure(figsize=(4, 2), dpi=180)
        try:
            ax = fig.add_axes([.1,.1,.8,.8]);ax.axis('off')
            table = ax.table(cellText=[[text]], colWidths=[.2], loc='center')
            table.auto_set_font_size(False);table.set_fontsize(9)
            table[(0,0)].set_height(height)
            with pytest.raises(ValueError, match='exceeds cell'):
                b.assert_text_inside_cells(fig, table)
        finally:
            plt.close(fig)


def test_impossible_single_glyph_is_explicit_failure(renderer):
    with pytest.raises(ValueError, match='single glyph'):
        b.wrapped('W', .01, renderer, FontProperties(size=9))


def test_tex_literal_breaks_do_not_remove_or_inject_visible_content():
    assert b.escape('A_B/C%&') == r'A\_\allowbreak{}B/\allowbreak{}C\%\&'
    assert b.escape(r'\{x}') == r'\textbackslash{}\{x\}'
    assert b.escape('123.45') == '123.45'


@pytest.mark.parametrize('token', ['0123456789abcdef'*4, 'W'*64, 'a'*12, 'a'*13])
def test_tex_long_literal_tokens_have_invisible_twelve_character_breaks(token):
    escaped = b.escape(token)
    chunks = escaped.split(r'\allowbreak{}')
    assert ''.join(chunks) == token
    assert all(0 < len(chunk) <= 12 for chunk in chunks)
    assert not escaped.endswith(r'\allowbreak{}')


def test_tex_breaks_do_not_split_escape_commands_and_preserve_delimiters():
    text = 'a'*12 + '%&' + '_SHA/' + 'b'*13 + '\nnext'
    expected = ('a'*12 + r'\%\&\_SHA/' + 'b'*13 + '\nnext')
    assert b.escape(text).replace(r'\allowbreak{}', '') == expected
    assert b.escape('count\u2013range\u2014note') == 'count--range---'+r'\allowbreak{}'+'note'


def test_tex_parent_and_note_sha_cells_wrap_without_changing_observations(tmp_path, monkeypatch):
    sha = '0123456789abcdef'*4
    frame = pd.DataFrame({'parent': [sha], 'note': ['registry '+sha]})
    before = frame.copy(deep=True)
    columns = pd.DataFrame({'column': ['parent','note'], 'label': ['Parent','Note'], 'width': [1,1]})
    monkeypatch.setattr(b, 'ROOT', tmp_path)
    b.tex_table(frame, columns, {'title':'Synthetic SHA cells', 'asset_id':'synthetic', 'caption':'Literal contents retained.'})
    tex = (tmp_path/'table.tex').read_text()
    expected = r'\allowbreak{}'.join(sha[i:i+12] for i in range(0,len(sha),12))
    assert tex.count(expected) == 2
    assert sha not in tex
    assert tex.replace(r'\allowbreak{}','').count(sha) == 2
    pd.testing.assert_frame_equal(frame, before)


def test_synthetic_standalone_table_renders_without_changing_rows(tmp_path, monkeypatch):
    columns = pd.DataFrame(dict(column=['case','metric','origin'],
        label=['Recorded regions / TRAIN target met','Full B WMAPE (%)','Status counts / label origins'],
        width=[1.2,.5,1.1]))
    data = pd.DataFrame([
        dict(case='AC_s03_FFTT_open / SURROGATE',metric='9.1836',origin='NEW_DETAILED_REGIONAL_FW_ON_FROZEN_P'),
        dict(case='AC_s03_ICNN_open / SURROGATE',metric='1.8830',origin='UNBROKEN_'+('W'*90)),
    ])
    original = data.copy(deep=True)
    spec = pd.DataFrame([dict(asset_id='synthetic',title='Synthetic wide headers',
                             caption='Synthetic observations only; no campaign data.')])
    frames = {'asset.csv':spec,'columns.csv':columns,'table.csv':data}
    monkeypatch.setattr(b,'ROOT',tmp_path)
    monkeypatch.setattr(b,'READS',{})
    def read(name):
        b.READS[name] = dict(file=name,sha256='synthetic',rows=len(frames[name]))
        return frames[name].copy(deep=True)
    monkeypatch.setattr(b,'read',read)
    monkeypatch.setattr(b,'csv_logical_names',lambda *args:[],raising=False)
    b.main()
    pd.testing.assert_frame_equal(data,original)
    result = json.loads((tmp_path/'RENDER_RESULT.json').read_text())
    assert result['status']=='RENDERED' and result['table_rows']==2
    assert all((tmp_path/name).is_file() for name in result['outputs'])
    tex=(tmp_path/'table.tex').read_text()
    assert all(b.escape(value) in tex for row in original.itertuples(index=False,name=None) for value in row)
    assert (tmp_path/'table.png').stat().st_size > 1000
