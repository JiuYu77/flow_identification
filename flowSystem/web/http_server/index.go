package http_server

import (
	"html/template"
	"net/http"
)

var (
	templates = template.Must(template.ParseFiles(
		"views/index.html",
		"views/threejs_head.html",
	))
)

func renderTemplate(w http.ResponseWriter, tmpl string, data interface{}) {
	err := templates.ExecuteTemplate(w, tmpl, data)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
	}
}

func Index(w http.ResponseWriter, r *http.Request) {
	renderTemplate(w, "index.html", nil)
}
