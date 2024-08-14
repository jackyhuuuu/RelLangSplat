from typing import Dict, Any, Callable, List, Tuple, NamedTuple, Text, Optional
from spacy.tokens.token import Token
from spacy.tokens.span import Span
from .heuristics import Heuristics

Rel = Tuple[List[Token], "Entity"]
Sup = List[Token]

DEFAULT_HEURISTICS = Heuristics()


def find_superlatives(tokens, heuristics) -> List[Sup]:
    """Modify and return a list of superlative tokens."""
    for heuristic in heuristics.superlatives:
        if any(tok.text in heuristic.keywords for tok in tokens):
            tokens.sort(key=lambda tok: tok.i)
            return [tokens]
    return []

def expand_chunks(doc, chunks):
    expanded = {}
    for key in chunks:
        chunk = chunks[key]
        start = chunk.start
        end = chunk.end
        for i in range(chunk.start-1, -1, -1):
            if any(doc[j].is_ancestor(doc[i]) for j in range(chunk.start, chunk.end)):
                if not any(any(doc[i].is_ancestor(doc[j]) for j in range(chunks[key2].start, chunks[key2].end)) for key2 in chunks if key != key2):
                    start = i
        for i in range(chunk.end, len(doc)):
            if any(doc[j].is_ancestor(doc[i]) for j in range(chunk.start, chunk.end)):
                if not any(any(doc[i].is_ancestor(doc[j]) or i == j for j in range(chunks[key2].start, chunks[key2].end)) for key2 in chunks if key != key2):
                    end = i+1
                else:
                    break
        expanded[key] = Span(doc=doc, start=start, end=end)
    return expanded

class Entity(NamedTuple):
    """Represents an entity with locative constraints extracted from the parse."""

    head: Span
    relations: List[Rel]
    superlatives: List[Sup]

    @classmethod
    def extract(cls, head, chunks, heuristics: Optional[Heuristics] = None) -> "Entity":
        """Extract entities from a spacy parse.

        Jointly recursive with `_get_rel_sups`."""
        # 如果没有传入启发式方法，则使用默认的启发式方法
        if heuristics is None:
            heuristics = DEFAULT_HEURISTICS

        # 如果头结点 head 不在 chunks 中，表示当前的头结点可能是谓语情况
        # 然后检查其子节点中是否有在 chunks 中的节点，如果有，将 head 更新为该子节点
        if head.i not in chunks:
            # Handles predicative cases.
            children = list(head.children)
            if children and children[0].i in chunks:
                head = children[0]
                # TODO: Also extract predicative relations.
            else:
                return None
        # 从 chunks 中获取头结点对应的块信息
        hchunk = chunks[head.i]
        # 调用 _get_rel_sups 方法，递归提取关系（relations）和 superlative
        rels, sups = cls._get_rel_sups(head, head, [], chunks, heuristics)
        return cls(hchunk, rels, sups)

    @classmethod
    def _get_rel_sups(cls, token, head, tokens, chunks, heuristics) -> Tuple[List[Rel], List[Sup]]:
        hchunk = chunks[head.i]
        is_keyword = any(token.text in h.keywords for h in heuristics.relations)
        is_keyword |= token.text in heuristics.null_keywords

        # Found another entity head.
        if token.i in chunks and chunks[token.i] is not hchunk and not is_keyword:
            tchunk = chunks[token.i]
            tokens.sort(key=lambda tok: tok.i)
            subhead = cls.extract(token, chunks, heuristics)
            return [(tokens, subhead)], []

        # End of a chain of modifiers.
        n_children = len(list(token.children))
        if n_children == 0:
            return [], find_superlatives(tokens + [token], heuristics)

        relations = []
        superlatives = []
        is_keyword |= any(token.text in h.keywords for h in heuristics.superlatives)
        for child in token.children:
            if token.i in chunks and child.i in chunks and chunks[token.i] is chunks[child.i]:
                if not any(child.text in h.keywords for h in heuristics.superlatives):
                    if n_children == 1:
                        # Catches "the goat on the left"
                        sups = find_superlatives(tokens + [token], heuristics)
                        superlatives.extend(sups)
                    continue
            new_tokens = tokens + [token] if token.i not in chunks or is_keyword else tokens
            subrel, subsup = cls._get_rel_sups(child, head, new_tokens, chunks, heuristics)
            relations.extend(subrel)
            superlatives.extend(subsup)
        return relations, superlatives

    def expand(self, span: Span = None):
        tokens = [token for token in self.head]
        if span is None:
            span = [None]
        for target_token in span:
            include = False
            stack = [token for token in self.head]
            while len(stack) > 0:
                token = stack.pop()
                if token == target_token:
                    token2 = target_token.head
                    while token2.head != token2:
                        tokens.append(token2)
                        token2 = token2.head
                    tokens.append(token2)
                    stack = []
                    include = True
                if target_token is None or include:
                    tokens.append(token)
                for child in token.children:
                    stack.append(child)
        tokens = list(set(tokens))
        tokens = sorted(tokens, key=lambda x: x.i)
        return ' '.join([token.text for token in tokens])

    def __eq__(self, other: "Entity") -> bool:
        if self.text != other.text:
            return False
        if self.relations != other.relations:
            return False
        if self.superlatives != other.superlatives:
            return False
        return True

    @property
    def text(self) -> Text:
        """Get the text predicate associated with this entity."""
        return self.head.text
