from typing import Protocol

from vectorist.ir.models import IRDocument


class AdapterProtocol(Protocol):
    def parse(self, source_uri: str, content: str) -> IRDocument: ...
