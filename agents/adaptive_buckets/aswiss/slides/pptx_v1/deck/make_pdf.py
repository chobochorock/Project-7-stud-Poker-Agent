"""PDF of the deck through LibreOffice, without the extra space LibreOffice puts between Hangul and Latin text.

LibreOffice applies "spacing between Asian and non-Asian text" (ODF style:text-autospace) to every paragraph of an
imported PPTX, which shows up as gaps such as "행동 ." and "( 우연 )". PowerPoint does not need this. So: PPTX -> ODP,
switch text-autospace off in the ODP styles, ODP -> PDF.

Usage: python make_pdf.py DECK.pptx OUT.pdf [--soffice "soffice"]
"""
import argparse, os, re, shlex, shutil, subprocess, tempfile, zipfile

ap = argparse.ArgumentParser()
ap.add_argument('deck')
ap.add_argument('out')
ap.add_argument('--soffice', default='soffice', help='command that runs LibreOffice (default: soffice)')
a = ap.parse_args()



def chart_style(mo):
    """Chart text has no paragraph properties in the ODP, so the default (autospace on) applies; add one that is off."""
    st = mo.group(0)
    if '<style:text-properties' not in st:
        return st
    if '<style:paragraph-properties' in st:
        return st.replace('<style:paragraph-properties', '<style:paragraph-properties style:text-autospace="none"', 1)
    return st.replace('<style:text-properties', '<style:paragraph-properties style:text-autospace="none"/><style:text-properties', 1)


cmd = shlex.split(a.soffice)
with tempfile.TemporaryDirectory() as tmp:
    src = os.path.join(tmp, 'deck.pptx')
    shutil.copy(a.deck, src)
    subprocess.run(cmd + ['--headless', '--convert-to', 'odp', '--outdir', tmp, src], check=True, capture_output=True, timeout=600)
    odp, fixed = os.path.join(tmp, 'deck.odp'), os.path.join(tmp, 'fixed.odp')
    n = 0
    with zipfile.ZipFile(odp) as zin, zipfile.ZipFile(fixed, 'w') as zout:
        zout.writestr(zipfile.ZipInfo('mimetype'), zin.read('mimetype'), compress_type=zipfile.ZIP_STORED)
        for item in zin.infolist():
            if item.filename == 'mimetype':
                continue
            data = zin.read(item.filename)
            if item.filename.endswith('.xml'):
                text = data.decode('utf-8')
                n += text.count('text-autospace="ideograph-alpha"')
                text = text.replace('text-autospace="ideograph-alpha"', 'text-autospace="none"')
                if item.filename.startswith('Object ') and item.filename.endswith('/content.xml'):
                    text = re.sub(r'<style:style\b[^>]*>.*?</style:style>', chart_style, text, flags=re.S)
                data = text.encode('utf-8')
            zout.writestr(item, data, compress_type=zipfile.ZIP_DEFLATED)
    if n == 0:
        print('warning: no text-autospace attribute found; the PDF may keep the Hangul/Latin gaps')
    subprocess.run(cmd + ['--headless', '--convert-to', 'pdf', '--outdir', tmp, fixed], check=True, capture_output=True, timeout=600)
    shutil.copy(os.path.join(tmp, 'fixed.pdf'), a.out)
print('wrote', a.out)
