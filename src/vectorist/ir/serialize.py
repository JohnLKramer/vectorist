from vectorist.ir.models import IRDocument


def ir_to_json(doc: IRDocument) -> str:
    return doc.model_dump_json()


def ir_from_json(data: str) -> IRDocument:
    return IRDocument.model_validate_json(data)
