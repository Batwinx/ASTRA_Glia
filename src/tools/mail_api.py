# src/tools/mail_api.py
import json
from datetime import datetime

def mail_api(query: str) -> str:
    """
    Simuliert E-Mail-API (z.B. Gmail/Outlook).
    In Produktion: Ersetze durch echte API mit OAuth.
    """
    print(f"   📧 Mail-API wird aufgerufen für: {query}")
    
    # Simulierte E-Mail-Aktion
    mail_action = {
        "query": query,
        "action": "compose" if "schreiben" in query.lower() or "senden" in query.lower() else "check",
        "status": "simulated",
        "timestamp": datetime.now().isoformat(),
        "message": "E-Mail-Aktion wurde simuliert. In Produktion: Echte API-Integration.",
        "source": "Demo-Mail-API (ASTRA_Glia)"
    }
    
    return json.dumps(mail_action, indent=2, ensure_ascii=False)