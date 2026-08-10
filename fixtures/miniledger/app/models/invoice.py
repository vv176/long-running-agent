"""Invoice record. RELATIVE import two levels up."""
from dataclasses import dataclass, field

from ..utils.money import Money


@dataclass
class Invoice:
    id: str
    customer_id: str
    lines: list = field(default_factory=list)

    def total(self):
        out = Money(0)
        for line in self.lines:
            out = out + line
        return out
