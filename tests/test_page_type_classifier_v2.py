from __future__ import annotations

import pandas as pd

from src.econometrics_eda_v2.page_type_classifier import (
    PAGE_TYPE_FAMILY,
    classify_page_type_family,
    classify_page_type_v2,
)
from src.econometrics_eda_v2.url_features import build_source_url_features


def _cls(**kwargs):
    return classify_page_type_v2(**kwargs)


def test_contact_page():
    r = _cls(url="https://hospital.test/contact", title="ติดต่อเรา")
    assert r.page_type == "contact_page"
    assert r.family == "access_contact"


def test_appointment_page():
    r = _cls(url="https://hospital.test/appointment", title="นัดหมายแพทย์")
    assert r.page_type == "appointment_page"


def test_department_or_center_page():
    r = _cls(url="https://hospital.test/center/heart", title="ศูนย์หัวใจ")
    assert r.page_type == "department_or_center_page"


def test_doctor_profile():
    r = _cls(url="https://hospital.test/doctor/somchai", title="ประวัติแพทย์ นพ.สมชาย")
    assert r.page_type == "doctor_profile"


def test_article_health_info():
    r = _cls(url="https://hospital.test/blog/skin-care", title="บทความความรู้สุขภาพผิว")
    assert r.page_type == "article_health_info"


def test_disease_condition_page():
    r = _cls(url="https://hospital.test/disease/diabetes", title="โรคเบาหวาน อาการและวิธีรักษา")
    assert r.page_type == "disease_condition_page"


def test_pdf_document():
    r = _cls(url="https://hospital.test/files/guide.pdf", title="Download PDF")
    assert r.page_type == "pdf_document"
    assert r.family == "document"


def test_third_party_platform_page():
    r = _cls(url="https://apps.apple.com/th/app/example/id123", title="App Store")
    assert r.page_type == "third_party_platform_page"


def test_price_page_with_strong_evidence():
    r = _cls(
        url="https://hospital.test/packages/checkup",
        title="แพ็กเกจตรวจสุขภาพ ราคา",
        page_text="แพ็กเกจตรวจสุขภาพ ราคา 1,500 บาท ราคา 2,500 บาท",
        table_count=1,
    )
    assert r.page_type == "price_package_page"
    assert "price" in r.evidence.lower()


def test_body_only_generic_price_does_not_force_price_page():
    r = _cls(
        url="https://hospital.test/article/skin-care",
        title="บทความความรู้สุขภาพผิว",
        page_text="This article mentions cost and price once, but it is educational.",
    )
    assert r.page_type != "price_package_page"


def test_unknown_has_reason():
    r = _cls(url="https://example.test/random", title="", page_text="tiny")
    assert r.page_type == "unknown"
    assert r.unknown_reason


def test_family_mapping_covers_all_labels():
    for page_type in PAGE_TYPE_FAMILY:
        assert classify_page_type_family(page_type)


def test_domain_plot_label_uses_url_root_not_ai_label():
    rows = pd.DataFrame(
        [
            {
                "source_row_id": "s1",
                "source_url": "https://www.mahidol.ac.th/abc",
                "normalized_url": "https://mahidol.ac.th/abc",
                "source_domain": "Mahidol University",
            }
        ]
    )
    out, _ = build_source_url_features(rows)
    assert out.loc[0, "source_domain_ai_label"] == "Mahidol University"
    assert out.loc[0, "domain_raw_looks_like_label"] is True or bool(out.loc[0, "domain_raw_looks_like_label"])
    assert out.loc[0, "source_domain_host"] == "www.mahidol.ac.th"
    assert out.loc[0, "source_root_domain"] == "mahidol.ac.th"
    assert out.loc[0, "domain_plot_label"] == "mahidol.ac.th"
