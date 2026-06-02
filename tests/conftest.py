"""Pytest fixtures: small in-memory toy reference sets."""

from __future__ import annotations

from typing import List

import pytest

from supervdj.models import GeneReference


@pytest.fixture
def trbv_refs() -> List[GeneReference]:
    return [
        GeneReference("TRBV6-5", "TRB", "V", "F", "CASS", 0.05),
        GeneReference("TRBV20-1", "TRB", "V", "F", "CSAR", 0.05),
        GeneReference("TRBV29-1", "TRB", "V", "F", "CSVE", 0.03),
        GeneReference("TRBV30", "TRB", "V", "F", "CAWS", 0.02),
    ]


@pytest.fixture
def trbj_refs() -> List[GeneReference]:
    return [
        GeneReference("TRBJ2-7", "TRB", "J", "F", "SYEQYF", 0.13),
        GeneReference("TRBJ1-1", "TRB", "J", "F", "NTEAFF", 0.06),
        GeneReference("TRBJ2-1", "TRB", "J", "F", "SYNEQFF", 0.11),
    ]


@pytest.fixture
def trav_refs() -> List[GeneReference]:
    return [
        GeneReference("TRAV1-1", "TRA", "V", "F", "CAVR", 0.02),
        GeneReference("TRAV14DV4", "TRA", "V", "F", "CAMRE", 0.02),
        GeneReference("TRAV13-1", "TRA", "V", "F", "CAAS", 0.03),
    ]


@pytest.fixture
def traj_refs() -> List[GeneReference]:
    return [
        GeneReference("TRAJ33", "TRA", "J", "F", "SDSNYQLIW", 0.03),
        GeneReference("TRAJ42", "TRA", "J", "F", "NYGGSQGNLIF", 0.025),
        GeneReference("TRAJ12", "TRA", "J", "F", "DSSYKLIF", 0.024),
    ]
