from os import PathLike
from operator import itemgetter
from pathlib import Path
from string import ascii_lowercase
from typing import Union

import numpy as np
import pandas as pd
from flask import Flask, request



app = Flask("hangman-server")

class Dataset:

    def __init__(self, dirpath: Path, allowed_chars: str=ascii_lowercase):
        self.whitelist = allowed_chars
        self.wordlist = pd.Series()
        for fpath in dirpath.iterdir():
            self.add(fpath)
    
    def add(self, filepath: Union[PathLike, str]):
        df = pd.read_csv(filepath, sep=';', names=['word'], index_col=False,
                         engine='python', encoding='unicode_escape',
                         on_bad_lines='warn', memory_map=True)
        sr = df.word.dropna()
        # Only retain words with allowed characters
        sr = sr[sr.str.fullmatch(f"[{self.whitelist}]+")]
        sr = pd.concat([sr, sr.str.lower()], ignore_index=True)
        sr.drop_duplicates(inplace=True)
        sr = pd.concat([self.wordlist, sr], ignore_index=True)
        sr.sort_values(inplace=True, ignore_index=True)
        self.wordlist = sr
    
    def __call__(self, known: str, excluded_chars: str="") -> np.ndarray:
        sr = self.wordlist
        # Build expression to match remaining characters
        exclude_set = set(excluded_chars) | set(known) - {'_'}
        exclude_expr = ""
        offset = 0
        for char in sorted(exclude_set) + ['$']:
            if offset:
                prev_char = exclude_expr[-1]
                expected_ord = ord(prev_char) + offset
                if ord(char) != expected_ord:
                    if offset > 1:
                        if offset > 2:
                            exclude_expr += '-'
                        exclude_expr += chr(expected_ord - 1)
                    exclude_expr += char
                    offset = 0
            else:
                exclude_expr += char
            offset += 1
        exclude_expr = exclude_expr[:-1]
        char_expr = f"[^{exclude_expr}]" if exclude_expr else '.'
        match_expr = ""
        counter = 0
        # Build expression to match potential words
        for char in known + '$':
            if char == '_':
                counter += 1
            else:
                if counter:
                    match_expr += char_expr
                    if counter > 1:
                        match_expr += '{' + str(counter) + '}'
                    counter = 0
                match_expr += char
        # Get all candidate words
        sr = sr.loc[sr.str.fullmatch(match_expr)]
        if sr.empty:
            raise RuntimeError('unknown word')
        # Split into columns for each letter
        df = sr.str.split('', expand=True)
        df = df.iloc[:, 1:-1].rename(columns=lambda x: x - 1)
        # Drop columns for known letters
        df = df.loc[:, pd.Series(list(known)) == '_']
        # Get counts for remaining characters
        counts = pd.concat([
            col.value_counts() for _, col in df.iteritems()
        ], axis=1)
        counts.replace(np.nan, 0, inplace=True)
        counts /= df.size
        return counts.sort_index()

dset = Dataset(Path('./wordlists'), ascii_lowercase)


@app.route("/")
def index():
    g = itemgetter('known', 'excluded')
    known, excluded = g(request.args)
    try:
        counts = dset(known, excluded)
    except RuntimeError:
        counts = pd.Series()
    return {
        'chars': ''.join(counts.index),
        'counts': counts.to_numpy().tolist(),
    }


@app.after_request
def add_cors_headers(response):
    allowed_origin = ['http://localhost:8080']
    if request.origin in allowed_origin:
        response.access_control_allow_origin = request.origin
        response.access_control_allow_methods = ['GET']
        response.access_control_allow_headers = ['Content-Type', 'Cache-Control', 'Authorization']
        response.access_control_max_age = 86400
    return response