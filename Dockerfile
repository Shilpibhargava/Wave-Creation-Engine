# Define the starting image for our application. This will contain
# all of the base operating system components and dependencies needed
# for Python 3.8. This will pull from our internal Docker Hub mirror.
#ENV JAVA_HOME="foo"

FROM hub.docker.target.com/python:3.8

# Adds Pipfile and Pipefile.lock into the image in the /app folder.
COPY Pipfile* /app/

# Set the initial working directory for the image to the /app folder
WORKDIR /app

# Adds the file from a URL into the image, this is to be used
# by pip & pipenv to communicate while on the corporate network.
ADD http://browserconfig.target.com/tgt-certs/tgt-ca-bundle.crt /app

# pip & pipenv look at the REQUESTS_CA_BUNDLE environmental variable
# in order to trust external communications. If encountering an SSL
# CERTIFICATE_VERIFY_FAILED error, this is usually the missing piece.
ENV REQUESTS_CA_BUNDLE="/app/tgt-ca-bundle.crt"

# In conjunction with copying the Pipfile & Pipefile.lock above
# this will install the pipenv tool, and then install all of the
# defined dependencies for this application as part of the image.

RUN apt-get update
RUN apt-get install default-jdk -y


RUN pip install --upgrade pip pipenv && \
    pipenv install --deploy

# The app folder is copied towards the end of the image creation
# since it tends to change more often than the previous steps,
# and therefore we can take advantage of Docker's caching to speed
# up subsequent builds which leads to a faster build/rebuild cycle.
COPY app /app

# When Docker starts the image instance (as a container) it will look at
# either the ENTRYPOINT or CMD directives and run those applications.
# Their usage is nuanced and situational dependent, but for now
# ENTRYPOINT is sufficient.
ENTRYPOINT ["/app/entrypoint.sh"]

# Advertise to the end user & Docker, that this image will listen on
# port 8050.  This doesn't explicitly open the port up, but it's more
# of a suggestion to people on how to use the image.
EXPOSE 8050