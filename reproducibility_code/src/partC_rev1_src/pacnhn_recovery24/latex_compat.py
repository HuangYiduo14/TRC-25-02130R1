"""Add a portable typography entry point without editing preserved TeX members.

The wrapper maps only U+2011/U+2013/U+2014, whose missing-glyph events were
observed in unchanged introduction/bibliography text. It loads no packages,
changes no mathematical commands, and does not invoke a compiler.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import re

FILENAME = 'main_recovery_compat.tex'
SCHEMA = 'RECOVERY24_LATEX_DASH_COMPATIBILITY_V1'
MAPPINGS = {'2011': 'NONBREAKING_HYPHEN', '2013': 'EN_DASH', '2014': 'EM_DASH'}


def wrapper_text():
    """Deterministic UTF-8 LaTeX wrapper; literal Unicode occurs only in definitions.

XeTeX and LuaTeX permit active Unicode characters. Modern pdfLaTeX exposes
DeclareUnicodeCharacter through its UTF-8 kernel input layer. Older formats
fail explicitly rather than downloading a package or silently dropping text.
"""
    return r'''% Recovery typography entry point. Original main.tex remains byte-identical.
% Only U+2011 NON-BREAKING HYPHEN, U+2013 EN DASH, U+2014 EM DASH are mapped.
% Compile this file instead of main.tex. No additional package is loaded.
\protected\def\PACNHNRecoveryNBHyphen{\nobreak\hbox{-}\nobreak}
\protected\def\PACNHNRecoveryEnDash{\textendash{}}
\protected\def\PACNHNRecoveryEmDash{\textemdash{}}
\newif\ifPACNHNRecoveryUnicodeEngine
\ifdefined\XeTeXrevision
  \PACNHNRecoveryUnicodeEnginetrue
\fi
\ifdefined\directlua
  \PACNHNRecoveryUnicodeEnginetrue
\fi
\ifPACNHNRecoveryUnicodeEngine
  \catcode"2011=13\relax
  \catcode"2013=13\relax
  \catcode"2014=13\relax
  \protected\def‑{\PACNHNRecoveryNBHyphen}
  \protected\def–{\PACNHNRecoveryEnDash}
  \protected\def—{\PACNHNRecoveryEmDash}
\else
  \ifdefined\DeclareUnicodeCharacter
    \DeclareUnicodeCharacter{2011}{\PACNHNRecoveryNBHyphen}
    \DeclareUnicodeCharacter{2013}{\PACNHNRecoveryEnDash}
    \DeclareUnicodeCharacter{2014}{\PACNHNRecoveryEmDash}
  \else
    \errmessage{Recovery wrapper requires UTF-8 capable LaTeX; no package was loaded}
  \fi
\fi
\input{main.tex}
'''


def _sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def _bundle(bundle,output_root):
    root=Path(output_root).resolve(strict=True);raw=Path(bundle)
    resolved=raw.resolve(strict=True)
    if not root.is_dir() or not resolved.is_dir() or resolved==root or not resolved.is_relative_to(root):
        raise ValueError('Wrapper requires an existing NEW output subdirectory')
    # A reparse destination must not redirect this sole new write.
    target=raw/FILENAME
    if target.is_symlink() or target.exists():raise FileExistsError('Preserve prior compatibility wrapper')
    main=resolved/'main.tex'
    if main.is_symlink() or not main.is_file() or main.resolve()!=main:
        raise ValueError('A regular preserved bundle-root main.tex is required')
    return resolved,main


def write_wrapper(bundle, *, output_root, expected_main_sha256):
    """Exclusive write of one NEW file; no modification of main or other members."""
    if not isinstance(expected_main_sha256,str) or re.fullmatch('[0-9a-f]{64}',expected_main_sha256) is None:
        raise ValueError('Exact preserved main.tex SHA-256 is required')
    root,main=_bundle(bundle,output_root)
    if _sha(main)!=expected_main_sha256:raise ValueError('Preserved main.tex digest differs')
    data=wrapper_text().encode('utf-8');path=root/FILENAME
    with path.open('xb') as stream:stream.write(data)
    if _sha(main)!=expected_main_sha256:raise ValueError('Preserved main.tex changed during wrapper creation')
    digest=hashlib.sha256(data).hexdigest()
    if _sha(path)!=digest:raise ValueError('New wrapper bytes changed')
    return dict(schema=SCHEMA,status='CREATED_UNCOMPILED_TYPOGRAPHY_WRAPPER',path=str(path),sha256=digest,
        filename=FILENAME,output_stem=Path(FILENAME).stem,
        main=dict(path=str(main),sha256=expected_main_sha256),mappings=dict(MAPPINGS),
        helper_sha256=_sha(__file__),existing_members_modified=False,packages_added=[],
        compiler_executed=False,scope='ONLY_THREE_LITERAL_UNICODE_DASH_MAPPINGS')


def verify_wrapper(bundle, record, *, output_root):
    """Recheck exact wrapper, original main and current helper before/after build."""
    root=Path(output_root).resolve(strict=True);bundle=Path(bundle).resolve(strict=True)
    path=bundle/FILENAME;main=bundle/'main.tex'
    if bundle==root or not bundle.is_relative_to(root) or not bundle.is_dir():
        raise ValueError('Compatibility wrapper is outside NEW output subtree')
    if path.is_symlink() or main.is_symlink() or not path.is_file() or not main.is_file():
        raise ValueError('Wrapper/main must be regular bundle members')
    wanted=hashlib.sha256(wrapper_text().encode('utf-8')).hexdigest()
    if (record.get('schema')!=SCHEMA or record.get('status')!='CREATED_UNCOMPILED_TYPOGRAPHY_WRAPPER'
            or record.get('scope')!='ONLY_THREE_LITERAL_UNICODE_DASH_MAPPINGS'
            or record.get('path')!=str(path) or record.get('sha256')!=wanted
            or record.get('filename')!=FILENAME or record.get('output_stem')!=path.stem
            or record.get('mappings')!=MAPPINGS or record.get('helper_sha256')!=_sha(__file__)
            or record.get('existing_members_modified') is not False or record.get('packages_added')!=[]
            or record.get('compiler_executed') is not False
            or record.get('main',{}).get('path')!=str(main)
            or _sha(path)!=wanted or _sha(main)!=record['main'].get('sha256')):
        raise ValueError('Compatibility wrapper or preserved main binding differs')
    return path
