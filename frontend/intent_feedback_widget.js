/**
 * Intent Feedback Widget
 * Collects user feedback on intent classification
 * 
 * Usage:
 * <div id="intent-feedback-widget"></div>
 * <script src="intent_feedback_widget.js"></script>
 * <script>
 *   IntentFeedbackWidget.init({
 *     sessionId: 'your-session-id',
 *     userId: 'optional-user-id',
 *     apiBaseUrl: 'http://localhost:8000',
 *     position: 'bottom-right' // or 'bottom-left', 'top-right', 'top-left'
 *   });
 * </script>
 */

class IntentFeedbackWidget {
    constructor(config) {
        this.config = {
            sessionId: config.sessionId || this.generateSessionId(),
            userId: config.userId || null,
            apiBaseUrl: config.apiBaseUrl || '',
            position: config.position || 'bottom-right',
            autoShow: config.autoShow !== false,
            language: config.language || 'en' // 'en' or 'tr'
        };
        
        this.currentClassification = null;
        this.feedbackSubmitted = false;
        
        this.translations = {
            en: {
                title: 'Was this helpful?',
                yes: 'Yes',
                no: 'No',
                whatShouldItBe: 'What should it be?',
                submit: 'Submit',
                thanks: 'Thank you for your feedback!',
                placeholder: 'Select correct intent...'
            },
            tr: {
                title: 'Yardımcı oldu mu?',
                yes: 'Evet',
                no: 'Hayır',
                whatShouldItBe: 'Ne olmalıydı?',
                submit: 'Gönder',
                thanks: 'Geri bildiriminiz için teşekkürler!',
                placeholder: 'Doğru amacı seçin...'
            }
        };
        
        this.intents = [
            { value: 'restaurant', label_en: 'Restaurant', label_tr: 'Restoran' },
            { value: 'attraction', label_en: 'Attraction', label_tr: 'Gezilecek Yer' },
            { value: 'museum', label_en: 'Museum', label_tr: 'Müze' },
            { value: 'transportation', label_en: 'Transportation', label_tr: 'Ulaşım' },
            { value: 'accommodation', label_en: 'Hotel', label_tr: 'Konaklama' },
            { value: 'event', label_en: 'Event', label_tr: 'Etkinlik' },
            { value: 'shopping', label_en: 'Shopping', label_tr: 'Alışveriş' },
            { value: 'nightlife', label_en: 'Nightlife', label_tr: 'Gece Hayatı' },
            { value: 'weather', label_en: 'Weather', label_tr: 'Hava Durumu' },
            { value: 'route_planning', label_en: 'Route Planning', label_tr: 'Rota Planlama' },
            { value: 'local_tips', label_en: 'Local Tips', label_tr: 'Yerel Tavsiyeler' },
            { value: 'greeting', label_en: 'Greeting', label_tr: 'Selamlaşma' },
            { value: 'general', label_en: 'General Question', label_tr: 'Genel Soru' }
        ];
        
        this.createWidget();
    }
    
    generateSessionId() {
        return 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    }
    
    getText(key) {
        const lang = this.config.language;
        return this.translations[lang][key] || this.translations.en[key];
    }
    
    createWidget() {
        // Create widget container
        const widget = document.createElement('div');
        widget.id = 'intent-feedback-widget';
        widget.className = `intent-feedback-widget ${this.config.position}`;
        widget.style.display = 'none';
        
        widget.innerHTML = `
            <div class="ifb-container">
                <button class="ifb-close" onclick="IntentFeedbackWidget.instance.hide()">&times;</button>
                <div class="ifb-title">${this.getText('title')}</div>
                <div class="ifb-buttons">
                    <button class="ifb-btn ifb-btn-yes" onclick="IntentFeedbackWidget.instance.submitFeedback(true)">
                        👍 ${this.getText('yes')}
                    </button>
                    <button class="ifb-btn ifb-btn-no" onclick="IntentFeedbackWidget.instance.showCorrection()">
                        👎 ${this.getText('no')}
                    </button>
                </div>
                <div class="ifb-correction" style="display: none;">
                    <div class="ifb-correction-title">${this.getText('whatShouldItBe')}</div>
                    <select class="ifb-intent-select" id="ifb-intent-select">
                        <option value="">${this.getText('placeholder')}</option>
                    </select>
                    <button class="ifb-btn ifb-btn-submit" onclick="IntentFeedbackWidget.instance.submitCorrection()">
                        ${this.getText('submit')}
                    </button>
                </div>
                <div class="ifb-thanks" style="display: none;">
                    ${this.getText('thanks')}
                </div>
            </div>
        `;
        
        // Add styles
        const style = document.createElement('style');
        style.textContent = this.getStyles();
        document.head.appendChild(style);
        
        // Append to body
        document.body.appendChild(widget);
        
        // Populate intent options
        const select = widget.querySelector('#ifb-intent-select');
        const lang = this.config.language;
        this.intents.forEach(intent => {
            const option = document.createElement('option');
            option.value = intent.value;
            option.textContent = lang === 'tr' ? intent.label_tr : intent.label_en;
            select.appendChild(option);
        });
        
        this.widget = widget;
    }
    
    getStyles() {
        return `
            .intent-feedback-widget {
                position: fixed;
                z-index: 9999;
                max-width: 300px;
                animation: slideIn 0.3s ease-out;
            }
            
            .intent-feedback-widget.bottom-right {
                bottom: 20px;
                right: 20px;
            }
            
            .intent-feedback-widget.bottom-left {
                bottom: 20px;
                left: 20px;
            }
            
            .intent-feedback-widget.top-right {
                top: 20px;
                right: 20px;
            }
            
            .intent-feedback-widget.top-left {
                top: 20px;
                left: 20px;
            }
            
            @keyframes slideIn {
                from {
                    transform: translateY(20px);
                    opacity: 0;
                }
                to {
                    transform: translateY(0);
                    opacity: 1;
                }
            }
            
            .ifb-container {
                background: white;
                border-radius: 12px;
                padding: 20px;
                box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
                border: 1px solid #e0e0e0;
            }
            
            .ifb-close {
                position: absolute;
                top: 8px;
                right: 8px;
                background: none;
                border: none;
                font-size: 24px;
                cursor: pointer;
                color: #999;
                line-height: 1;
                padding: 0;
                width: 24px;
                height: 24px;
            }
            
            .ifb-close:hover {
                color: #333;
            }
            
            .ifb-title {
                font-size: 16px;
                font-weight: 600;
                margin-bottom: 15px;
                color: #333;
            }
            
            .ifb-buttons {
                display: flex;
                gap: 10px;
            }
            
            .ifb-btn {
                flex: 1;
                padding: 10px 16px;
                border: none;
                border-radius: 8px;
                font-size: 14px;
                font-weight: 500;
                cursor: pointer;
                transition: all 0.2s;
            }
            
            .ifb-btn-yes {
                background: #4CAF50;
                color: white;
            }
            
            .ifb-btn-yes:hover {
                background: #45a049;
                transform: scale(1.02);
            }
            
            .ifb-btn-no {
                background: #f44336;
                color: white;
            }
            
            .ifb-btn-no:hover {
                background: #da190b;
                transform: scale(1.02);
            }
            
            .ifb-correction {
                margin-top: 15px;
            }
            
            .ifb-correction-title {
                font-size: 14px;
                margin-bottom: 10px;
                color: #666;
            }
            
            .ifb-intent-select {
                width: 100%;
                padding: 10px;
                border: 1px solid #ddd;
                border-radius: 6px;
                font-size: 14px;
                margin-bottom: 10px;
            }
            
            .ifb-btn-submit {
                width: 100%;
                background: #2196F3;
                color: white;
            }
            
            .ifb-btn-submit:hover {
                background: #0b7dda;
                transform: scale(1.02);
            }
            
            .ifb-thanks {
                text-align: center;
                color: #4CAF50;
                font-weight: 500;
                padding: 10px 0;
            }
        `;
    }
    
    show(classificationData) {
        if (this.feedbackSubmitted) {
            return; // Don't show again if feedback already submitted
        }
        
        this.currentClassification = classificationData;
        this.widget.style.display = 'block';
        
        // Reset state
        this.widget.querySelector('.ifb-buttons').style.display = 'flex';
        this.widget.querySelector('.ifb-correction').style.display = 'none';
        this.widget.querySelector('.ifb-thanks').style.display = 'none';
    }
    
    hide() {
        this.widget.style.display = 'none';
    }
    
    showCorrection() {
        this.widget.querySelector('.ifb-buttons').style.display = 'none';
        this.widget.querySelector('.ifb-correction').style.display = 'block';
    }
    
    async submitFeedback(isCorrect) {
        try {
            const payload = {
                session_id: this.config.sessionId,
                user_id: this.config.userId,
                query: this.currentClassification.query,
                language: this.currentClassification.language,
                predicted_intent: this.currentClassification.intent,
                predicted_confidence: this.currentClassification.confidence,
                classification_method: this.currentClassification.method,
                latency_ms: this.currentClassification.latency_ms,
                is_correct: isCorrect,
                feedback_type: 'explicit'
            };
            
            const response = await fetch(`${this.config.apiBaseUrl}/api/feedback/intent`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(payload)
            });
            
            if (response.ok) {
                this.showThanks();
            } else {
                console.error('Failed to submit feedback:', await response.text());
            }
        } catch (error) {
            console.error('Error submitting feedback:', error);
        }
    }
    
    async submitCorrection() {
        const select = this.widget.querySelector('#ifb-intent-select');
        const actualIntent = select.value;
        
        if (!actualIntent) {
            alert(this.getText('placeholder'));
            return;
        }
        
        try {
            const payload = {
                session_id: this.config.sessionId,
                user_id: this.config.userId,
                query: this.currentClassification.query,
                language: this.currentClassification.language,
                predicted_intent: this.currentClassification.intent,
                predicted_confidence: this.currentClassification.confidence,
                classification_method: this.currentClassification.method,
                latency_ms: this.currentClassification.latency_ms,
                is_correct: false,
                actual_intent: actualIntent,
                feedback_type: 'explicit'
            };
            
            const response = await fetch(`${this.config.apiBaseUrl}/api/feedback/intent`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(payload)
            });
            
            if (response.ok) {
                this.showThanks();
            } else {
                console.error('Failed to submit correction:', await response.text());
            }
        } catch (error) {
            console.error('Error submitting correction:', error);
        }
    }
    
    showThanks() {
        this.widget.querySelector('.ifb-buttons').style.display = 'none';
        this.widget.querySelector('.ifb-correction').style.display = 'none';
        this.widget.querySelector('.ifb-thanks').style.display = 'block';
        
        this.feedbackSubmitted = true;
        
        // Auto-hide after 2 seconds
        setTimeout(() => {
            this.hide();
        }, 2000);
    }
    
    // Track implicit feedback
    async trackImplicitFeedback(userAction, timeSpentSeconds = null) {
        try {
            if (!this.currentClassification) return;
            
            const payload = {
                session_id: this.config.sessionId,
                query: this.currentClassification.query,
                predicted_intent: this.currentClassification.intent,
                predicted_confidence: this.currentClassification.confidence,
                user_action: userAction,
                time_spent_seconds: timeSpentSeconds
            };
            
            await fetch(`${this.config.apiBaseUrl}/api/feedback/intent/implicit`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(payload)
            });
        } catch (error) {
            console.error('Error tracking implicit feedback:', error);
        }
    }
    
    // Static method to initialize
    static init(config) {
        if (!IntentFeedbackWidget.instance) {
            IntentFeedbackWidget.instance = new IntentFeedbackWidget(config);
        }
        return IntentFeedbackWidget.instance;
    }
}

// Export for use in other scripts
if (typeof module !== 'undefined' && module.exports) {
    module.exports = IntentFeedbackWidget;
}
