#!/usr/bin/env python3
"""Test script for the abstract scraper.

Tests:
1. Publisher detection from URLs
2. Rule-based extraction for each publisher
3. Sequential batch fetching
4. LLM fallback (if configured)
"""

import logging
import sys
import time
from typing import Dict, List

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("test_scraper")

# Test DOIs for each publisher
TEST_DOIS: List[Dict[str, str]] = [
    {
        "doi": "10.1145/3503250",
        "title": "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis",
        "publisher": "ACM",
    },
    {
        "doi": "10.1016/j.neucom.2025.132095",
        "title": "Fake information spreaders detection",
        "publisher": "Elsevier",
    },
    {
        "doi": "10.3390/rs17030557",
        "title": "SAR Image Quantization",
        "publisher": "MDPI",
    },
    {
        "doi": "10.1117/12.2320024",
        "title": "SAR GAN paper",
        "publisher": "SPIE",
    },
    {
        "doi": "10.1080/01431161.2025.2593683",
        "title": "Taylor & Francis Test",
        "publisher": "Taylor & Francis",
    },
]


def test_publisher_detection():
    """Test publisher detection from URLs."""
    from zotwatch.infrastructure.enrichment import detect_publisher

    test_urls = [
        ("https://dl.acm.org/doi/10.1145/3503250", "acm"),
        ("https://ieeexplore.ieee.org/document/8099498", "ieee"),
        ("https://link.springer.com/article/10.1007/s00371-024-03546-y", "springer"),
        ("https://www.sciencedirect.com/science/article/pii/S0925231225000001", "elsevier"),
        ("https://www.spiedigitallibrary.org/conference-proceedings-of-spie/10647/2320024", "spie"),
        ("https://www.mdpi.com/2072-4292/17/3/557", "mdpi"),
        ("https://arxiv.org/abs/2003.08934", "arxiv"),
        ("https://www.nature.com/articles/s41586-024-07487-w", "springer"),
        ("https://www.tandfonline.com/doi/full/10.1080/01431161.2025.2593683", "taylor_francis"),
        ("https://unknown-publisher.com/paper", "unknown"),
    ]

    logger.info("=" * 60)
    logger.info("Testing publisher detection")
    logger.info("=" * 60)

    passed = 0
    for url, expected in test_urls:
        detected = detect_publisher(url)
        status = "PASS" if detected == expected else "FAIL"
        if status == "PASS":
            passed += 1
        logger.info("[%s] %s -> %s (expected: %s)", status, url[:50], detected, expected)

    logger.info("Publisher detection: %d/%d passed", passed, len(test_urls))
    return passed == len(test_urls)


def test_single_doi(doi: str, title: str = None, publisher: str = None):
    """Test fetching a single DOI."""
    from zotwatch.infrastructure.enrichment import AbstractScraper

    logger.info("-" * 60)
    logger.info("Testing DOI: %s (%s)", doi, publisher or "unknown")
    logger.info("-" * 60)

    # Create scraper without LLM (rules only)
    scraper = AbstractScraper(llm=None, use_llm_fallback=False)

    try:
        start_time = time.time()
        abstract = scraper.fetch_abstract(doi, title)
        elapsed = time.time() - start_time

        if abstract:
            logger.info("SUCCESS! Extracted abstract in %.1fs", elapsed)
            logger.info("Abstract length: %d chars", len(abstract))
            logger.info("Abstract preview: %s...", abstract[:200])
            return True
        else:
            logger.warning("FAILED: No abstract found (%.1fs)", elapsed)
            return False

    except Exception as e:
        logger.error("ERROR: %s", e)
        import traceback

        traceback.print_exc()
        return False

    finally:
        scraper.close()


def test_batch_sequential():
    """Test sequential batch fetching."""
    from zotwatch.infrastructure.enrichment import AbstractScraper

    logger.info("=" * 60)
    logger.info("Testing batch fetching (%d DOIs)", len(TEST_DOIS))
    logger.info("=" * 60)

    scraper = AbstractScraper(llm=None, use_llm_fallback=False)

    try:
        start_time = time.time()
        results = scraper.fetch_batch(TEST_DOIS)
        elapsed = time.time() - start_time

        logger.info("Batch complete: %d/%d in %.1fs", len(results), len(TEST_DOIS), elapsed)
        for doi, abstract in results.items():
            logger.info("  %s: %d chars", doi, len(abstract))

        return results

    finally:
        scraper.close()


def main():
    """Run all tests."""
    logger.info("=" * 60)
    logger.info("Abstract Scraper Test Suite")
    logger.info("=" * 60)

    # Test 1: Publisher detection
    detection_ok = test_publisher_detection()
    logger.info("")

    # Test 2: Single DOI tests (one from each publisher)
    single_results = {}
    for item in TEST_DOIS[:2]:  # Test first 2 DOIs individually
        doi = item["doi"]
        success = test_single_doi(doi, item.get("title"), item.get("publisher"))
        single_results[doi] = success
        logger.info("")

    # Test 3: Batch test
    logger.info("")
    batch_results = test_batch_sequential()

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    logger.info("Publisher detection: %s", "PASS" if detection_ok else "FAIL")
    logger.info("Single DOI tests: %d/%d passed", sum(single_results.values()), len(single_results))
    logger.info("Batch fetch: %d/%d abstracts extracted", len(batch_results), len(TEST_DOIS))
    logger.info("=" * 60)

    # Return exit code
    all_passed = detection_ok and all(single_results.values()) and len(batch_results) >= len(TEST_DOIS) // 2
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
