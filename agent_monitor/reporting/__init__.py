from . import alerts, dashboard

try:
    from . import pdf
except ImportError:
    pdf = None  # optional: requires reportlab
