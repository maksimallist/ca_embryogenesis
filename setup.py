import os
import re

from setuptools import setup, find_packages

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

meta_path = os.path.join(__location__, '_meta.py')
with open(meta_path) as meta:
    exec(meta.read())


def read_requirements():
    """parses requirements from requirements.txt"""
    requirements_path = os.path.join(__location__, 'requirements.txt')
    with open(requirements_path, encoding='utf8') as f:
        requirements = [line.strip() for line in f if not line.strip().startswith('#')]

    names = []
    links = []
    for req in requirements:
        if '://' in req:
            links.append(req)
        else:
            names.append(req)

    return {'install_requires': names, 'dependency_links': links}


def readme():
    with open(os.path.join(__location__, 'README.md'), encoding='utf8') as f:
        text = f.read()
    text = re.sub(r']\((?!https?://)', r'](https://github.com/maksimallist/ca_embryogenesis/master', text)
    # text = re.sub(r'\ssrc="(?!https?://)', r' src="https://raw.githubusercontent.com/deepmipt/DeepPavlov/master/', text)
    return text


if __name__ == '__main__':
    setup(
        name='ca_embryogenesis',
        packages=find_packages(exclude=('tests', 'docs', 'utils')),
        version=__version__,
        description=__description__,
        long_description=readme(),
        long_description_content_type='text/markdown',
        author=__author__,
        author_email=__email__,
        license=__license__,
        url='https://github.com/maksimallist/ca_embryogenesis',
        # download_url=f'https://github.com/deepmipt/DeepPavlov/archive/{__version__}.tar.gz',
        keywords=__keywords__,
        include_package_data=True,
        **read_requirements()
    )
