"""Source-preservation and explicit-engine fixtures; no compiler is launched."""
import hashlib
from pathlib import Path
import re

import pytest
from partC_rev1_src.pacnhn_recovery24 import latex_compat as c


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


@pytest.fixture
def bundle(tmp_path):
    root=tmp_path/'new';path=root/'publication/bundle';path.mkdir(parents=True)
    (path/'main.tex').write_text('\\documentclass{article}\n\\input{introduction}\n',encoding='utf-8')
    (path/'introduction.tex').write_text('high‑performance simulation–assignment predictions—sometimes',encoding='utf-8')
    (path/'cas-refs.bib').write_text('title={road–rail}',encoding='utf-8')
    return root,path


def test_exact_three_characters_and_latex_replacements():
    text=c.wrapper_text()
    assert {ord(x) for x in text if ord(x)>127}=={0x2011,0x2013,0x2014}
    assert re.findall(r'\\catcode"([0-9A-F]+)=13',text)==['2011','2013','2014']
    assert re.findall(r'\\DeclareUnicodeCharacter\{([0-9A-F]+)\}',text)==['2011','2013','2014']
    assert r'\nobreak\hbox{-}\nobreak' in text
    assert r'\textendash{}' in text and r'\textemdash{}' in text
    assert '2212' not in text and r'\mathcode' not in text


def test_engine_branches_no_packages_or_engine_specific_execution():
    text=c.wrapper_text()
    for name in ('XeTeXrevision','directlua','DeclareUnicodeCharacter'):assert '\\ifdefined\\'+name in text
    assert r'\directlua{' not in text
    assert r'\RequirePackage' not in text and r'\usepackage' not in text
    assert r'\errmessage' in text
    assert text.endswith('\\input{main.tex}\n') and text.count(r'\input{')==1
    assert r'\documentclass' not in text and r'\font' not in text


def test_active_character_definition_is_scanned_after_catcode():
    text=c.wrapper_text()
    for code,char in [('2011','‑'),('2013','–'),('2014','—')]:
        assert text.index('\\catcode"'+code)<text.index('\\protected\\def'+char)
    conditional_lines=[line.strip() for line in text.splitlines()]
    assert sum(line.startswith((r'\ifdefined',r'\ifPACNHNRecoveryUnicodeEngine')) for line in conditional_lines)==sum(line==r'\fi' for line in conditional_lines)


def test_new_file_only_original_members_byte_exact(bundle):
    root,path=bundle;before={p.name:p.read_bytes() for p in path.iterdir()}
    result=c.write_wrapper(path,output_root=root,expected_main_sha256=digest(path/'main.tex'))
    assert {p.name for p in path.iterdir()}==set(before)|{c.FILENAME}
    assert all((path/name).read_bytes()==raw for name,raw in before.items())
    assert (path/c.FILENAME).read_bytes()==c.wrapper_text().encode('utf-8')
    assert c.verify_wrapper(path,result,output_root=root)==path/c.FILENAME
    assert result['compiler_executed'] is False and not result['existing_members_modified']


def test_repeat_fails_preserving_existing_wrapper(bundle):
    root,path=bundle;result=c.write_wrapper(path,output_root=root,expected_main_sha256=digest(path/'main.tex'))
    before=(path/c.FILENAME).read_bytes()
    with pytest.raises(FileExistsError):c.write_wrapper(path,output_root=root,expected_main_sha256=result['main']['sha256'])
    assert (path/c.FILENAME).read_bytes()==before


@pytest.mark.parametrize('value',[None,'','f'*63,'g'*64,'f'*64])
def test_wrong_or_missing_main_binding_precedes_write(bundle,value):
    root,path=bundle
    with pytest.raises(ValueError):c.write_wrapper(path,output_root=root,expected_main_sha256=value)
    assert not (path/c.FILENAME).exists()


def test_outside_or_root_destination_rejected(bundle,tmp_path):
    root,path=bundle;outside=tmp_path/'old';outside.mkdir();(outside/'main.tex').write_bytes(b'original')
    with pytest.raises(ValueError):c.write_wrapper(outside,output_root=root,expected_main_sha256=digest(outside/'main.tex'))
    with pytest.raises(ValueError):c.write_wrapper(root,output_root=root,expected_main_sha256='a'*64)
    assert not (outside/c.FILENAME).exists()


@pytest.mark.parametrize('member',['main.tex','main_recovery_compat.tex'])
def test_post_creation_member_change_rejected(bundle,member):
    root,path=bundle;result=c.write_wrapper(path,output_root=root,expected_main_sha256=digest(path/'main.tex'))
    with (path/member).open('ab') as stream:stream.write(b'%changed')
    with pytest.raises(ValueError):c.verify_wrapper(path,result,output_root=root)


@pytest.mark.parametrize('field,value',[('helper_sha256','a'*64),('mappings',{}),('filename','main.tex'),
    ('output_stem','main'),('existing_members_modified',True),('packages_added',['inputenc']),('compiler_executed',True),
    ('status','COMPILED'),('scope','ALL_UNICODE')])
def test_record_cannot_reinterpret_scope(bundle,field,value):
    root,path=bundle;result=c.write_wrapper(path,output_root=root,expected_main_sha256=digest(path/'main.tex'));result[field]=value
    with pytest.raises(ValueError):c.verify_wrapper(path,result,output_root=root)


def test_missing_main_is_not_generated(bundle):
    root,path=bundle;(path/'main.tex').unlink()
    with pytest.raises(ValueError):c.write_wrapper(path,output_root=root,expected_main_sha256='a'*64)
    assert not (path/c.FILENAME).exists()
