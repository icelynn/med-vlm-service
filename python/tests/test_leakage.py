# -*- coding: utf-8 -*-
"""Offline test for build_rsna_pool.scan_safe_studies -- the structural
leakage guard the R2 image-retrieval result depends on (a study containing
even one of the 150 RSNA eval image IDs must be dropped in its entirety, not
just the matching slice; see core-challenges #12/#0h). No network or Kaggle
credentials needed: the Kaggle API client and the paginated file-listing call
are both replaced with fakes.

Run:  ./.venv/Scripts/python.exe python/tests/test_leakage.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "rag"))

import build_rsna_pool as pool  # noqa: E402


class _FakeFile:
    def __init__(self, name):
        self.name = name


class _FakePage:
    def __init__(self, files, next_page_token):
        self.dataset_files = files
        self.next_page_token = next_page_token


class _FakeKaggleApi:
    def authenticate(self):
        pass


def _build_fake_pages():
    """Three studies designed to exercise both leakage-guard paths:
    - study 100: fully safe, single page -> every image must survive.
    - study 200: one excluded id, single page -> the WHOLE study must drop,
      including the non-matching id (ID_eee).
    - study 300: split across the page boundary, with the excluded id only
      appearing on the SECOND page -- this is the case the docstring claims
      to guard against ("a study is only finalized once the NEXT study's
      rows start"); a premature finalize on page 1 alone would wrongly let
      ID_fff leak into the pool before ID_ggg's exclusion is even seen.
    """
    study_100 = [
        "pngs/pngs/100/series1/IM_0000-ID_aaa.png",
        "pngs/pngs/100/series1/IM_0001-ID_bbb.png",
        "pngs/pngs/100/series1/IM_0002-ID_ccc.png",
    ]
    study_200 = [
        "pngs/pngs/200/series1/IM_0000-ID_ddd.png",  # excluded
        "pngs/pngs/200/series1/IM_0001-ID_eee.png",  # not excluded itself, but same study
    ]
    study_300_page1 = ["pngs/pngs/300/series1/IM_0000-ID_fff.png"]
    study_300_page2 = ["pngs/pngs/300/series1/IM_0001-ID_ggg.png"]  # excluded

    page1 = _FakePage([_FakeFile(n) for n in study_100 + study_200 + study_300_page1], "token2")
    page2 = _FakePage([_FakeFile(n) for n in study_300_page2], None)
    return {None: page1, "token2": page2}


def test_excluded_study_dropped_wholesale_and_page_boundary_grouping():
    fake_pages = _build_fake_pages()

    def _fake_list_files_with_retry(api, token, page_size, max_retries=5, base_delay=10):
        return fake_pages[token]

    original_api, original_list = pool.KaggleApi, pool._list_files_with_retry
    pool.KaggleApi = _FakeKaggleApi
    pool._list_files_with_retry = _fake_list_files_with_retry
    try:
        excluded = {"ID_ddd", "ID_ggg"}
        safe_images = pool.scan_safe_studies(excluded, max_pages=10)
    finally:
        pool.KaggleApi = original_api
        pool._list_files_with_retry = original_list

    assert set(safe_images) == {"ID_aaa", "ID_bbb", "ID_ccc"}, (
        f"expected only study 100's 3 images to survive, got {sorted(safe_images)}"
    )
    assert "ID_ddd" not in safe_images and "ID_eee" not in safe_images, (
        "study 200 must be dropped wholesale, including the non-matching ID_eee"
    )
    assert "ID_fff" not in safe_images and "ID_ggg" not in safe_images, (
        "study 300 (split across the page boundary) must still be dropped wholesale -- "
        "ID_fff must not leak just because its exclusion (ID_ggg) arrived on the next page"
    )


