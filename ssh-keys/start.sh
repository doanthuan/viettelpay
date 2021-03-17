#!/bin/sh
# Extract the certs
#openssl pkcs12 -in DoiTac-Poc.p12 -out viettel.crt.pem -clcerts -nokeys
#openssl pkcs12 -in DoiTac-Poc.p12 -out viettel.key.pem -nocerts -nodes

# Connect to the VPN
sudo openfortivpn 117.4.245.203:9999 --username=DoiTac-Poc --trusted-cert 3b487e58072c5e371e996746c8c0f2a60fbba80b5bd3bc0498eeed206b7ca4de --user-cert viettel.crt.pem --user-key viettel.key.pem
