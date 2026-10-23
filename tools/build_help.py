"""Render the versioned offline help; use --check to reject stale bundled pages."""

import argparse
from collections import Counter
from html import escape
from html.parser import HTMLParser
import json
from pathlib import Path
import posixpath
import re
import tomllib
from urllib.parse import quote, unquote, urlsplit

import mistune
from latex2mathml.converter import convert


ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / 'desktop/packed_bed_ui/assets/help'
SCREENS = ('landing', 'project', 'general', 'chemistry', 'bed', 'program', 'report', 'results', 'studies')
GUIDES = {
    'docs/GUI-GUIDE.md': 'gui-guide.html',
    'docs/EQUATIONS.md': 'equations.html',
    'docs/PLUGINS.md': 'plugins.html',
    'docs/CLI-GUIDE.md': 'cli-guide.html',
    'examples/plugins/README.md': 'plugin-examples.html',
    'packed_bed/kinetics/KINETICS_SOURCES.md': 'kinetics.html',
    'docs/help/installation.md': 'installation.html',
    'THIRD_PARTY_NOTICES.md': 'notices.html',
}
PLUGINS = {
    'plugins/waste_iron/README.md': 'waste-iron.html',
}
SOURCES = {
    'docs/help/getting-started.md': 'getting-started.html',
    **{f'docs/help/{name}.md': f'{name}.html' for name in SCREENS},
    **GUIDES,
    **PLUGINS,
}


class PageParser(HTMLParser):
    """Collect rendered text and local targets, including section search entries."""

    def __init__(self, body):
        super().__init__(convert_charrefs=True)
        self.ids, self.links, self.text = set(), [], []
        self.sections = []
        self.section = None
        self.feed(body)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if 'id' in attrs:
            if attrs['id'] in self.ids:
                raise ValueError(f"Duplicate HTML id: {attrs['id']}")
            self.ids.add(attrs['id'])
        if tag == 'a':
            self.links.append(attrs.get('href', ''))
        if tag in {'h1', 'h2', 'h3', 'h4'}:
            self.section = {'anchor': attrs.get('id', ''), 'text': []}
            self.sections.append(self.section)

    def handle_data(self, data):
        self.text.append(data)
        if self.section is not None:
            self.section['text'].append(data)


def plain(text):
    return re.sub(r'\s+', ' ', ''.join(PageParser(text).text)).strip()


def render_math(text, display):
    markup = convert(text, display=display)
    # latex2mathml retains TeX's underscore escape inside text labels.
    return re.sub(r'(<mtext\b[^>]*>)(.*?)(</mtext>)',
                  lambda m: m[1] + m[2].replace(r'\_', '_') + m[3], markup, flags=re.S)


class HelpRenderer(mistune.HTMLRenderer):
    def __init__(self, source, source_ref):
        super().__init__(escape=True)
        self.source = source
        self.source_ref = source_ref
        self.headings = []
        self.slugs = Counter()

    def heading(self, text, level, **attrs):
        title = plain(text)
        slug = re.sub(r'[^\w\- ]', '', title.lower()).replace(' ', '-')
        count = self.slugs[slug]
        self.slugs[slug] += 1
        anchor = f'{slug}-{count}' if count else slug
        self.headings.append((level, title, anchor))
        return f'<h{level} id="{escape(anchor)}">{text}</h{level}>\n'

    def link(self, text, url, title=None):
        parts = urlsplit(url)
        external = bool(parts.scheme or parts.netloc)
        if not external and parts.path:
            source = posixpath.normpath(posixpath.join(posixpath.dirname(self.source), unquote(parts.path)))
            if source == 'README.md' and parts.fragment == 'documentation':
                url = 'index.html'
            elif source in SOURCES:
                url = SOURCES[source] + (f'#{parts.fragment}' if parts.fragment else '')
            else:
                # Source listings and downloadable examples stay explicit online links.
                if not (ROOT / source).exists():
                    raise ValueError(f'{self.source}: missing link target {source}')
                kind = 'tree' if (ROOT / source).is_dir() else 'blob'
                url = f'https://github.com/Griminir5/multisolid-CL/{kind}/{quote(self.source_ref, safe="")}/{quote(source)}'
                if parts.fragment:
                    url += '#' + parts.fragment
                external = True
        if external:
            title = (title + ' — ' if title else '') + 'Online resource; internet connection required'
            text += ' <span class="online">(online)</span>'
        return super().link(text, url, title)

    def inline_math(self, text):
        return render_math(text, 'inline')

    def block_math(self, text):
        return '<div class="equation">' + render_math(text, 'block') + '</div>\n'


def navigation(titles, current):
    def item(filename):
        active = ' aria-current="page"' if filename == current else ''
        return f'<a href="{filename}"{active}>{escape(titles[filename])}</a>'

    result = item('index.html') + item('getting-started.html')
    for label, filenames in (
        ('Screen help', [f'{name}.html' for name in SCREENS]),
        ('Guides and reference', list(GUIDES.values())),
        ('Plugins', list(PLUGINS.values())),
    ):
        opened = ' open' if current in filenames else ''
        result += f'<details{opened}><summary>{label}</summary>'
        result += ''.join(item(filename) for filename in filenames) + '</details>'
    return result


def build_pages():
    version = tomllib.loads((ROOT / 'desktop/pyproject.toml').read_text(encoding='utf-8'))['project']['version']
    pages, titles, headings = {}, {'index.html': 'MultiSolid help'}, {}
    for source, filename in SOURCES.items():
        renderer = HelpRenderer(source, f'v{version}')
        markdown = mistune.create_markdown(renderer=renderer, plugins=['table', 'strikethrough', 'url', 'math'])
        body = markdown((ROOT / source).read_text(encoding='utf-8'))
        pages[filename] = body
        headings[filename] = renderer.headings
        titles[filename] = renderer.headings[0][1]

    cards = ''.join(f'<li><a href="{name}.html">{escape(titles[name + ".html"])}</a></li>' for name in SCREENS)
    plugins = ''.join(f'<li><a href="{filename}">{escape(titles[filename])}</a></li>'
                      for filename in PLUGINS.values())
    pages['index.html'] = (
        '<h1 id="multisolid-help">MultiSolid help</h1>'
        '<p>Find your next step, understand the screen you are using, or explore the model and reference guides.</p>'
        '<p><a class="start" href="getting-started.html">Getting started →</a></p>'
        '<h2 id="screen-help">Help for your screen</h2>'
        '<p>Press F1 in MultiSolid to open the page for the active screen or case tab.</p>'
        f'<ul class="cards">{cards}</ul>'
        '<h2 id="guides">Guides and reference</h2>'
        '<ul><li><a href="installation.html">Install the desktop application</a> — downloads and portable installations.</li>'
        '<li><a href="gui-guide.html">User guide</a> — projects, cases, runs, reports, and studies.</li>'
        '<li><a href="equations.html">Model equations</a> — assumptions, balances, and numerical treatment.</li>'
        '<li><a href="plugins.html">Plugin guide</a> and <a href="plugin-examples.html">authoring examples</a>.</li>'
        '<li><a href="cli-guide.html">CLI guide</a> — YAML inputs, commands, and Python API.</li>'
        '<li><a href="kinetics.html">Kinetics sources</a> — scientific references for built-in models.</li>'
        '<li><a href="gui-guide.html#troubleshooting">Troubleshooting</a> — common problems and checks.</li></ul>'
        '<h2 id="plugins">Plugins</h2>'
        '<p>Documentation for official scientific plugins.</p>'
        f'<ul>{plugins}</ul>'
        '<p>These pages and their equations work offline. Links marked “online” open external resources.</p>'
    )
    headings['index.html'] = [(2, 'Help for your screen', 'screen-help'), (2, 'Guides and reference', 'guides'),
                              (2, 'Plugins', 'plugins')]
    parsed = {name: PageParser(body) for name, body in pages.items()}
    for name, page in parsed.items():
        for link in page.links:
            parts = urlsplit(link)
            if parts.scheme or parts.netloc:
                continue
            target = parts.path or name
            if target not in parsed or (parts.fragment and unquote(parts.fragment) not in parsed[target].ids):
                raise ValueError(f'{name}: broken help link {link}')

    output, search = {}, []
    template = (ROOT / 'tools/help/template.html').read_text(encoding='utf-8')
    for filename, body in pages.items():
        toc = ''.join(f'<a class="level-{level}" href="#{escape(anchor)}">{escape(title)}</a>'
                      for level, title, anchor in headings[filename] if level in (2, 3))
        output[filename] = template.format(title=escape(titles[filename]), version=escape(version),
                                           navigation=navigation(titles, filename), toc=toc, body=body)
        section_titles = {anchor: title for _, title, anchor in headings[filename]}
        for section in parsed[filename].sections:
            search.append({'url': filename + '#' + section['anchor'], 'page': titles[filename],
                           'title': section_titles.get(section['anchor'], titles[filename]),
                           'text': re.sub(r'\s+', ' ', ' '.join(section['text'])).strip()})
    output['search-data.js'] = 'window.helpSearch = ' + json.dumps(search, ensure_ascii=False) + ';\n'
    for name in ('help.css', 'help.js'):
        output[name] = (ROOT / 'tools/help' / name).read_text(encoding='utf-8')
    output['manifest.json'] = json.dumps({'version': version, 'pages': titles}, indent=2, ensure_ascii=False) + '\n'
    return output


def build(destination=DESTINATION, *, check=False):
    output = build_pages()
    changed = [name for name, text in output.items()
               if not (destination / name).is_file() or (destination / name).read_bytes() != text.encode('utf-8')]
    extra = [path.name for path in destination.iterdir() if path.is_file() and path.name not in output] if destination.exists() else []
    if check:
        if changed or extra:
            raise ValueError('Bundled help is stale; run python tools/build_help.py: ' + ', '.join(changed + extra))
    else:
        destination.mkdir(parents=True, exist_ok=True)
        for name, text in output.items():
            (destination / name).write_bytes(text.encode('utf-8'))
        for name in extra:
            (destination / name).unlink()
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    build(check=args.check)
    print('Offline help is up to date.' if args.check else f'Built offline help in {DESTINATION}')
