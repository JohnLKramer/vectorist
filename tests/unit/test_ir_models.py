from vectorist.ir.models import (
    CodeBlockNode,
    IRDocument,
    NodeType,
    ParagraphNode,
    SectionNode,
)
from vectorist.ir.serialize import ir_from_json, ir_to_json


def test_round_trip_ir_document():
    para = ParagraphNode(
        node_id="p1",
        parent_id="s1",
        children_ids=[],
        breadcrumbs=["doc.md", "Intro"],
        line_range=(3, 4),
        raw_content="Hello",
        metadata={},
    )
    section = SectionNode(
        node_id="s1",
        parent_id=None,
        children_ids=["p1"],
        breadcrumbs=["doc.md", "Intro"],
        line_range=(1, 4),
        raw_content="Intro",
        metadata={},
        level=1,
        title="Intro",
    )
    code = CodeBlockNode(
        node_id="c1",
        parent_id="s1",
        children_ids=[],
        breadcrumbs=["doc.md", "Intro"],
        line_range=(5, 10),
        raw_content="print(1)",
        metadata={},
        language="python",
        detection_source="fence",
    )
    section.children_ids = ["p1", "c1"]
    doc = IRDocument(
        source_uri="doc.md",
        root_id="s1",
        nodes={n.node_id: n for n in (section, para, code)},
    )
    restored = ir_from_json(ir_to_json(doc))
    assert restored.source_uri == "doc.md"
    assert restored.nodes["c1"].node_type == NodeType.CODE_BLOCK
    assert restored.nodes["c1"].language == "python"
