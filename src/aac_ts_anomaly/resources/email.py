##### Sending messages:
#################################################

import smtplib, email, ssl

smtp_server = "tmu.mail.allianz"
port = 25 

password = input("Type your password and press enter: ")

# Create a secure SSL context
context = ssl.create_default_context()

# Try to log in to server and send email
try:
    server = smtplib.SMTP(smtp_server,port)
    server.connect(smtp_server,port)
    #server.ehlo() # Can be omitted
    server.starttls(context=context) # Secure the connection
    #server.login(sender_email, password)   # not needed in AGN!
except Exception as e:
    print(e)

################ Send Mail ########################
print("\nSending Mail..........")

message = email.message.EmailMessage()

message.set_default_type("text/plain")
message["From"] = "alexander.vosseler1@allianz.de"     
message["To"] = ["alexander.vosseler1@allianz.de", "alexander.hoeweler@allianz.de"]
message["Subject"] =  "Python test email"

body = '''
Dear colleague,

How are you doing? This is a Python-Bot generated test email.

Regards,
Boty
'''

message.set_content(body)

response = server.send_message(msg=message)

print("\nLogging Out....")
resp_code, response = server.quit()
# END

    

#SMTP_SERVER = "outlook.office365.com"
#SMTP_SERVER = "smtp-mail.outlook.com"
#SMTP_SERVER = "mail.allianz.de"
#SMTP_SERVER = "de001-surf.zone2.proxy.allianz:8080@outlook.office365.com"
SMTP_SERVER = "tmu.mail.allianz"    # Gateway??

#my_proxy = "http://de001-surf.zone2.proxy.allianz"
#proxy_port = 25
#---------------------------------------------------------------------
#'proxy_port' should be an integer
#'PROXY_TYPE_SOCKS4' can be replaced to HTTP or PROXY_TYPE_SOCKS5
#socks.setdefaultproxy(socks.HTTP, my_proxy, proxy_port)
#socks.wrapmodule(smtplib)
#-------------------------------------------------------------------------


################# SMTP SSL ################################
start = time.time()
try:
    context = ssl.create_default_context()
    smtp_ssl = smtplib.SMTP_SSL(host=SMTP_SERVER, port=25, context=context)      # 465
except Exception as e:
    print(e) ;
    #print("ErrorType : {}, Error : {}".format(type(e).__name__, e))
    smtp_ssl = None

print("Connection Object : {}".format(smtp_ssl))
print("Total Time Taken  : {:,.2f} Seconds".format(time.time() - start))

######### Log In to mail account ############################
print("\nLogging In.....")  
resp_code, response = smtp_ssl.login(user = email_user, password = email_pass)

print("Response Code : {}".format(resp_code))
print("Response      : {}".format(response.decode()))

################ Send Mail ########################
print("\nSending Mail..........")

message = email.message.EmailMessage()

message.set_default_type("text/plain")
message["From"] = "a.vosseler@gmx.net"     
message["To"] = ["alexandervosseler@gmail.com", "a.vosseler@gmx.net"]
message["Subject"] =  "Test Email"

body = '''
Hello dear colleague,

How are you doing? This is a bot generated test email.

Regards,
Alex
'''

message.set_content(body)

response = smtp_ssl.send_message(msg=message)

print("List of Failed Recipients : {}".format(response))

######### Log out to mail account ############################
print("\nLogging Out....")
resp_code, response = smtp_ssl.quit()

print("Response Code : {}".format(resp_code))
print("Response      : {}".format(response.decode()))
