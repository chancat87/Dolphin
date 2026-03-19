from os import PathLike
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Union

T = Union[str, bytes]


def _read_symbol_table(symbol_table_file):
    symbol_table = {}
    with open(symbol_table_file, 'r', encoding='utf8') as fin:
        for line in fin:
            arr = line.strip().split()
            assert len(arr) == 2
            symbol_table[arr[0]] = int(arr[1])
    return symbol_table


class BaseTokenizer(ABC):

    def detokenize(self, ids: List[int]) -> Tuple[str, List[T]]:
        tokens = self.ids2tokens(ids)
        text = self.tokens2text(tokens)
        return text, tokens

    @abstractmethod
    def tokens2text(self, tokens: List[T]) -> str:
        raise NotImplementedError("abstract method")

    @abstractmethod
    def tokens2ids(self, tokens: List[T]) -> List[int]:
        raise NotImplementedError("abstract method")

    @abstractmethod
    def ids2tokens(self, ids: List[int]) -> List[T]:
        raise NotImplementedError("abstract method")

    @abstractmethod
    def vocab_size(self) -> int:
        raise NotImplementedError("abstract method")

    @property
    @abstractmethod
    def symbol_table(self) -> Dict[T, int]:
        raise NotImplementedError("abstract method")


class CharTokenizer(BaseTokenizer):

    def __init__(
        self,
        symbol_table: PathLike,
        connect_symbol: str = '',
        unk='<unk>',
    ) -> None:
        self._symbol_table = _read_symbol_table(symbol_table)
        self.char_dict = {v: k for k, v in self._symbol_table.items()}
        self.connect_symbol = connect_symbol
        self.unk = unk

    def tokens2text(self, tokens: List[str]) -> str:
        return self.connect_symbol.join(tokens)

    def tokens2ids(self, tokens: List[str]) -> List[int]:
        ids = []
        for ch in tokens:
            if ch in self._symbol_table:
                ids.append(self._symbol_table[ch])
            elif self.unk in self._symbol_table:
                ids.append(self._symbol_table[self.unk])
        return ids

    def ids2tokens(self, ids: List[int]) -> List[str]:
        content = [self.char_dict[w] for w in ids]
        return content

    def vocab_size(self) -> int:
        return len(self.char_dict)

    @property
    def symbol_table(self) -> Dict[str, int]:
        return self._symbol_table


class BpeTokenizer(CharTokenizer):

    def __init__(
        self,
        bpe_model: Union[PathLike, str],
        symbol_table: PathLike,
    ) -> None:
        super().__init__(symbol_table=symbol_table)
        self._model = bpe_model
        self.sp = None

    def _build_sp(self):
        if self.sp is None:
            import sentencepiece as spm
            self.sp = spm.SentencePieceProcessor()
            self.sp.load(self._model)

    def tokens2text(self, tokens: List[str]) -> str:
        self._build_sp()
        return self.sp.DecodePieces(list(tokens))


def init_tokenizer(configs) -> BaseTokenizer:
    tokenizer_type = configs.get("tokenizer", "char")
    tokenizer_conf = configs["tokenizer_conf"]

    if tokenizer_type == "char":
        tokenizer = CharTokenizer(
            symbol_table=tokenizer_conf['symbol_table_path'],
            connect_symbol=tokenizer_conf.get('connect_symbol', '')
        )
    elif tokenizer_type == "bpe":
        tokenizer = BpeTokenizer(
            bpe_model=tokenizer_conf["bpe_path"],
            symbol_table=tokenizer_conf["symbol_table_path"]
        )
    else:
        assert False, f"{tokenizer_type} is not supported!"

    return tokenizer
