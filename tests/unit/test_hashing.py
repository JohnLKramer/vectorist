from vectorist.hashing import sha256_text


def test_sha256_text_is_stable_hex():
    h1 = sha256_text("# Hello\n")
    h2 = sha256_text("# Hello\n")
    assert h1 == h2
    assert len(h1) == 64
    assert h1 != sha256_text("# Hello\n ")
