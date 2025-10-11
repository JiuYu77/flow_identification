package main

import "web/http_server"

func main() {
	h := http_server.Router(":8080")
	h.Run(nil)

	// h := http_server.Router(":8080", "cert.pem", "cert.key")
	// h.RunTLS(nil)
}
