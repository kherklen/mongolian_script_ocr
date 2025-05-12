#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Generate synthetic dataset from Mongolian song lyrics using the program 'pango-view'."""
__author__ = 'Erdene-Ochir Tuguldur'

import os
import gzip
import random

fonts = ['CMs Urga']
font_sizes = [29, 30, 31, 32]
punctuations = ['᠀', '᠀᠋', '᠀᠌', '᠀᠍'] + ['᠂', '᠃', '᠅', '》', '《', '?']
numerals = list('᠐᠑᠒᠓᠔᠕᠖᠗᠘᠙')

def read_content(f='../bichig2cyrillic/lyrics.txt.gz', max_words=10):
    lines = []
    for line in gzip.open(f, 'rt', encoding='utf-8').readlines():
        if '|' not in line:
            continue
        _, traditional_txt = line.strip().split('|')
        if len(traditional_txt.split()) <= max_words:
            lines.append(traditional_txt)
    return lines

def inject_punctuations_and_numerals(text):
    words = text.split()
    
    # Add punctuation more frequently
    if random.random() < 0.5:
        punctuation = random.choice(punctuations)
        if random.random() < 0.5:
            text = punctuation + ' ' + text
        else:
            text = text + punctuation

    # Add numerals in between words (increase frequency if needed)
    if random.random() < 0.3 and len(words) >= 2:
        insert_idx = random.randint(1, len(words) - 1)
        numeral = random.choice(numerals)
        words.insert(insert_idx, numeral)
        text = ' '.join(words)

    return text

def generate_image(text, output_file):
    font = random.choice(fonts)
    font_size = random.choice(font_sizes)
    cmd = f"pango-view --output='{output_file}' --rotate=-90 --no-display --font='{font} {font_size}' --margin='5 0 5 0' --text \"{text}\""
    os.system(cmd)

if __name__ == '__main__':
    os.makedirs('images3', exist_ok=True)
    for idx, val in enumerate(read_content()):
        if idx >= 1000:
            break
        val = inject_punctuations_and_numerals(val)
        output_file = f'images3/lyrics-{idx}.png'
        generate_image(val, output_file)
        print(f'{output_file}|{val}')
