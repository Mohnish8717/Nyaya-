"""Routing regressions: the route decides which machine answers."""
from __future__ import annotations

from app import routing


def test_emergency_beats_everything() -> None:
    route = routing.route_turn("my husband is beating me and he is coming now, please help")
    assert route.name == "emergency"
    assert route.emergency is True
    assert route.confidence >= 0.9


def test_procedure_match_for_a_known_process() -> None:
    route = routing.route_turn("how do I file an FIR at the police station")
    assert route.name == "procedural"
    assert route.procedure["id"] == "police-fir"


def test_statutory_question_is_not_hijacked_by_a_procedure() -> None:
    route = routing.route_turn("what does section 138 of the Negotiable Instruments Act say")
    assert route.name == "statute"


def test_precedent_request() -> None:
    route = routing.route_turn("find a similar judgment where the high court held that a deposit must be returned")
    assert route.name == "precedent"


def test_draft_request() -> None:
    route = routing.route_turn("write a notice to my landlord about the deposit")
    assert route.name == "draft"


def test_handoff_request() -> None:
    route = routing.route_turn("I want to talk to a lawyer, not a bot")
    assert route.name == "human"
    assert route.wants_human is True


def test_a_statement_of_the_problem_can_name_a_process_without_a_cue() -> None:
    """Most people do not say "how do I". A statement that matches a curated
    card by its own trigger words still reaches the card."""
    wages = routing.route_turn("kitna paisa bacha hai company ne salary nahi di")
    assert wages.name == "procedural"
    assert wages.procedure["id"] == "unpaid-wages"

    fraud = routing.route_turn("Someone called for OTP and the money was debited from my bank account just now")
    assert fraud.name == "procedural"
    assert fraud.procedure["id"] == "cybercrime-report"


def test_a_cue_family_still_beats_a_bare_signal() -> None:
    statute = routing.route_turn("what does section 138 of the Negotiable Instruments Act say")
    assert statute.name == "statute"
    drafted = routing.route_turn("write a notice to my landlord about the deposit")
    assert drafted.name == "draft"


def test_the_three_cards_that_were_missing_now_exist() -> None:
    """These were `unclear` because no card covered the situation, which looked
    like a routing bug and was a catalogue gap."""
    lockout = routing.route_turn("my landlord changed the lock and threw out my belongings")
    assert lockout.name == "procedural"
    assert lockout.procedure["id"] == "illegal-eviction-lockout"

    deposit = routing.route_turn("landlord will not return my deposit after I vacated the flat")
    assert deposit.name == "procedural"
    assert deposit.procedure["id"] == "tenancy-deposit-refund"

    custody = routing.route_turn("police arrested my brother last night and nobody will tell us where he is")
    assert custody.name == "procedural"
    assert custody.procedure["id"] == "arrest-rights"


def test_a_deposit_at_a_bank_is_not_a_tenancy() -> None:
    route = routing.route_turn("what is the interest rate on a bank fixed deposit")
    assert not route.procedure or route.procedure["id"] != "tenancy-deposit-refund"


def test_unknown_stays_unknown() -> None:
    route = routing.route_turn("hello")
    assert route.name in {"unclear", "human"}
    assert route.confidence < 0.9


def test_every_route_name_is_declared() -> None:
    assert set(routing.ROUTES) >= {"emergency", "human", "procedural", "statute", "precedent", "draft", "unclear"}
