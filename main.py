@app.post("/respond", response_model=RespondOut)
def respond(body: RespondIn):
    s = get_session(body.session_id)

    intro_done = bool(s.get("intro_done"))
    text = (body.text or "").strip().lower()

    # ---------- INTRO FLOW ----------
    if not intro_done:
        intro_chunks = fetch_chunks(
            s["board"],
            s["class_name"],
            s["subject"],
            s["chapter"],
            "INTRO",
            limit=50,
        )

        intro_idx = safe_int(s.get("chunk_index"), 0)

        # first call: serve intro chunk
        if not text:
            if intro_idx < len(intro_chunks):
                chunk_text = (intro_chunks[intro_idx].get("text", "") or "").strip()
                update_session(body.session_id, {
                    "stage": "INTRO",
                    "chunk_index": intro_idx + 1
                })
                return RespondOut(
                    ok=True,
                    session_id=body.session_id,
                    stage="INTRO",
                    teacher_text=chunk_text,
                    action="WAIT_FOR_STUDENT",
                    meta={"kind": "INTRO", "idx": intro_idx + 1},
                )

            return RespondOut(
                ok=True,
                session_id=body.session_id,
                stage="INTRO",
                teacher_text="Hi! I’m your GurukulAI teacher 😊 What’s your name?",
                action="WAIT_FOR_STUDENT",
                meta={},
            )

        # student says yes -> move to teaching
        if "yes" in text:
            teach_chunks = fetch_chunks(
                s["board"],
                s["class_name"],
                s["subject"],
                s["chapter"],
                "TEACH",
                limit=200,
            )

            update_session(body.session_id, {
                "intro_done": True,
                "stage": "TEACHING",
                "chunk_index": 0
            })

            if teach_chunks:
                first_text = (teach_chunks[0].get("text", "") or "").strip()
                update_session(body.session_id, {"chunk_index": 1})
                return RespondOut(
                    ok=True,
                    session_id=body.session_id,
                    stage="TEACHING",
                    teacher_text=first_text,
                    action="SPEAK",
                    meta={"kind": "TEACH", "idx": 1},
                )

            return RespondOut(
                ok=True,
                session_id=body.session_id,
                stage="TEACHING",
                teacher_text="Great! But I could not find teaching chunks for this chapter.",
                action="NOOP",
                meta={},
            )

        # treat other text as student name
        update_session(body.session_id, {"student_name": body.text.strip()})
        return RespondOut(
            ok=True,
            session_id=body.session_id,
            stage="INTRO",
            teacher_text=f"Nice to meet you, {body.text.strip()} 😊 When you are ready, say YES.",
            action="WAIT_FOR_STUDENT",
            meta={},
        )

    # ---------- TEACHING FLOW ----------
    teach_chunks = fetch_chunks(
        s["board"],
        s["class_name"],
        s["subject"],
        s["chapter"],
        "TEACH",
        limit=200,
    )

    idx = safe_int(s.get("chunk_index"), 0)

    if idx >= len(teach_chunks):
        return RespondOut(
            ok=True,
            session_id=body.session_id,
            stage="TEACHING",
            teacher_text="Chapter done ✅ Want a quiz now?",
            action="CHAPTER_DONE",
            meta={"done": True},
        )

    chunk_text = (teach_chunks[idx].get("text", "") or "").strip()
    update_session(body.session_id, {"chunk_index": idx + 1})

    return RespondOut(
        ok=True,
        session_id=body.session_id,
        stage="TEACHING",
        teacher_text=chunk_text if chunk_text else "Let’s continue…",
        action="SPEAK",
        meta={"kind": "TEACH", "idx": idx + 1},
    )
