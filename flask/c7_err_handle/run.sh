export FLASK_APP=main.py
export FLASK_DEBUG=1

# # to test the email notification
# export FLASK_DEBUG=0
# python -m smtpd -n -c DebuggingServer localhost:8025 &
# export MAIL_SERVER=localhost
# export MAIL_PORT=8025

flask run
