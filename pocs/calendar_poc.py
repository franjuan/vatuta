from __future__ import annotations

import datetime as dt
import os.path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# Permiso solo-lectura de Google Calendar
SCOPES = ["https://www.googleapis.com/auth/calendar.readonly"]  # :contentReference[oaicite:5]{index=5}


def get_service():
    creds = None
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
            creds = flow.run_local_server(port=0)
        with open("token.json", "w") as token:
            token.write(creds.to_json())

    service = build("calendar", "v3", credentials=creds)
    return service


def list_events(
    calendar_id: str = "primary",
    time_min: str | None = None,
    time_max: str | None = None,
    max_results: int = 2500,
):
    service = get_service()

    # Si no pasas rango, coge los próximos 30 días
    if not time_min:
        time_min = dt.datetime.utcnow().isoformat() + "Z"
    if not time_max:
        time_max = (dt.datetime.utcnow() + dt.timedelta(days=30)).isoformat() + "Z"

    events_result = (
        service.events()
        .list(
            calendarId=calendar_id,  # "primary" o el ID de otro calendario
            timeMin=time_min,
            timeMax=time_max,
            singleEvents=True,  # expande eventos recurrentes
            orderBy="startTime",
            maxResults=max_results,
        )
        .execute()
    )

    events = events_result.get("items", [])
    return events


if __name__ == "__main__":
    # Ejemplo: del 1 al 30 del mes actual (UTC)
    now = dt.datetime.utcnow()
    start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    if now.month == 12:
        end = start.replace(year=now.year + 1, month=1)
    else:
        end = start.replace(month=now.month + 1)

    events = list_events(
        calendar_id="primary",
        time_min=start.isoformat() + "Z",
        time_max=end.isoformat() + "Z",
    )

    if not events:
        print("No hay eventos en el rango.")
    else:
        for ev in events:
            start_time = ev.get("start", {}).get("dateTime") or ev.get("start", {}).get("date")
            end_time = ev.get("end", {}).get("dateTime") or ev.get("end", {}).get("date")
            summary = ev.get("summary", "(Sin título)")
            print(f"{start_time} → {end_time} | {summary}")
