@app.post("/ai/feedback")
async def submit_chat_feedback(request: Request, data: dict, db: Session = Depends(get_db)):
    """Submit feedback (like/dislike) for AI chat responses"""
    try:
        session_id = data.get("session_id")
        feedback_type = data.get("feedback_type")  # "like" or "dislike"
        user_query = data.get("user_query", "")
        ai_response = data.get("ai_response", "")
        message_index = data.get("message_index", 0)
        
        if not session_id or feedback_type not in ["like", "dislike"]:
            raise HTTPException(status_code=400, detail="Invalid feedback data")
        
        client_ip = getattr(request.client, 'host', 'unknown')
        
        # Create feedback record
        feedback_record = UserFeedback(
            session_id=session_id,
            feedback_type=feedback_type,
            user_query=user_query,
            response_preview=ai_response[:200] if ai_response else "",
            message_content=ai_response,
            message_index=message_index,
            user_ip=client_ip,
            timestamp=datetime.utcnow()
        )
        
        db.add(feedback_record)
        db.commit()
        
        return {
            "success": True,
            "message": f"Feedback ({feedback_type}) submitted successfully",
            "feedback_id": feedback_record.id
        }
        
    except Exception as e:
        db.rollback()
        logger.error(f"Error submitting chat feedback: {e}")
        raise HTTPException(status_code=500, detail="Failed to submit feedback")
