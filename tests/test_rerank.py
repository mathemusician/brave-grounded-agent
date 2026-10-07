from rerank import rrf, tokenize


def test_tokenize_unescapes_markdown_so_error_codes_match():
    assert "err_ssl_protocol_error" in tokenize(r"How to fix ERR\_SSL\_PROTOCOL\_ERROR in Chrome")
    assert tokenize("ERR_SSL_PROTOCOL_ERROR") == ["err_ssl_protocol_error"]


def test_rrf_puts_item_found_by_both_rankers_first():
    assert rrf([0, 1], [1, 2])[0] == 1


def test_rrf_includes_items_from_either_ranking():
    assert sorted(rrf([0, 1], [2, 3])) == [0, 1, 2, 3]
