from itertools import chain, count
from pathlib import Path
from string import ascii_lowercase
from typing import Iterable, Union

import numpy as np
from peewee import *
from playhouse.sqlite_ext import SqliteExtDatabase
from tqdm import tqdm


DATABASE_PATH = 'state.db'

_ELLIPSES = '.' * 3

class Dataset:

    def __init__(self, ngram_path: str=None, db_path: str=DATABASE_PATH,
                 allowed_chars: str=ascii_lowercase, max_row_count: int=1000):

        self.database = SqliteExtDatabase(db_path, regexp_function=True)
        self.allowed_chars = frozenset(allowed_chars)
        self.max_row_count = max_row_count

        class BaseModel(Model):
            class Meta:
                database = self.database

        class Word(BaseModel):
            chars = CharField(unique=True)
            freq = IntegerField(default=0)
        
        class File(BaseModel):
            path = CharField(unique=True)
            updated = TimestampField(default=0)
        
        self.Word = Word
        self.File = File

        self.create_tables()

        if ngram_path is not None:
            self.load_ngrams(ngram_path)
    
    def __call__(self, known: str, excluded_chars: str="") -> np.ndarray:
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
        match_expr = "^"
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
        char_idx = {c: i for c, i in zip(self.allowed_chars, count())}
        counts = np.zeros((len(char_idx), len(known)))
        # Get all candidate words
        with self.database:
            rows = self.Word.select().where(self.Word.chars.regexp(match_expr))
        for i, word in enumerate(rows):
            for j, c in enumerate(word.chars):
                counts[char_idx[c], j] += word.freq
            if i >= self.max_row_count:
                break
        counts /= counts.sum()
        return counts
    
    def create_tables(self):

        with self.database:
            self.database.create_tables([self.Word, self.File])
    
    def load_ngrams(self, path: str):

        print("Loading data from", path)

        with self.database:
            with tqdm(desc="Scanning files", position=0, leave=False) as pbar:
                files = []
                for p in self._find_files(Path(path)):
                    start = pbar.n
                    try:
                        with p.open('r') as f:
                            for _ in f:
                                pbar.update()
                    except BaseException:
                        continue
                    abs_path = str(p.resolve())
                    mtime = int(p.stat().st_mtime)
                    row = self.File.get_or_create(path=abs_path)[0]
                    if mtime > row.updated:
                        files.append((p, row, mtime, pbar.n - start))
                total = pbar.n

            with tqdm(total=total, position=0) as pbar:
                for i, (p, row, mtime, total) in enumerate(files):
                    desc = f"File {i + 1} of {len(files)}"
                    pbar.set_description(desc.ljust(16))
                    try:
                        self._load_ngrams_from_file(p, total, pbar)
                    except ValueError:
                        pass
                    row.updated = mtime
                    row.save()
    
    def _load_ngrams_from_file(self, path: Path, total: int,
                               pbar: tqdm, print_every: int=64):

        with path.open('r') as f:
            with tqdm(f, total=total, leave=False, position=1) as fpbar:

                for i, line in enumerate(fpbar):
                    parts = line.rstrip('\n').split('\t')
                    word = parts[0].split('_')[0].casefold()
                    if i % print_every == 0:
                        desc = pad_or_truncate(word)
                        fpbar.set_description(desc)
                    if not frozenset(word) <= self.allowed_chars:
                        continue
                    try:
                        freq = sum(int(p.split(',')[1]) for p in parts[1:])
                    except IndexError:
                        raise ValueError("expected Version 3 ngram format")

                    row = self.Word.get_or_create(chars=word)[0]
                    row.freq += freq
                    row.save()

                    pbar.update()

    def _find_files(self, path: Union[Path, Iterable[Path]]) -> Iterable[Path]:
        if isinstance(path, str):
            path = Path(path)
        elif isinstance(path, Iterable):
            return chain.from_iterable(map(self._find_files, path))
        if path.is_file():
            return (path,)
        elif path.is_dir():
            return self._find_files(path.iterdir())


def pad_or_truncate(str_: str, length: int=16, trunc: str='right') -> str:
    str_ = str_.ljust(length)
    over = len(str_) - length
    if over > 0:
        over += 3
        if trunc == 'left':
            str_ = _ELLIPSES + str_[over:]
        elif trunc == 'right':
            str_ = str_[:-over] + _ELLIPSES
        else:
            raise ValueError("argument 'trunc' must be given "
                             "as either 'left' or 'right'")
    return str_
