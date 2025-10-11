package http_server

import (
	"log"
	"net/http"
	"slices"
)

type Http struct {
	addr       string
	certFile   string
	keyFile    string
	validPaths []string // 记录所有注册的路径，用于检查请求路径是否合法
}

func NewHttp(addr string, opts ...string) *Http {
	var h = &Http{addr: addr}
	if len(opts) == 2 {
		h.certFile = opts[0]
		h.keyFile = opts[1]
	}
	return h
}

// false 表示路径不存在，不合法
func (h Http) contains(path string) bool {
	return slices.Contains(h.validPaths, path)
}

func (h *Http) registerPath(path string) {
	if !h.contains(path) { // 避免重复注册
		h.validPaths = append(h.validPaths, path)
	}
}

func (h Http) makeHandler(method string, handler http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != method {
			w.WriteHeader(http.StatusMethodNotAllowed)
			LogNotAllowed(method, r.URL.Path) // log
			return
		}
		if !h.contains(r.URL.Path) { // 检查请求路径是否合法
			w.WriteHeader(http.StatusNotFound)
			LogNotFound(method, r.URL.Path) // log
			return
		}

		LogFound(method, r.URL.Path) // log
		handler(w, r)
	}
}

func (h *Http) GET(relativePath string, handler http.HandlerFunc) {
	h.registerPath(relativePath)

	handler = h.makeHandler(http.MethodGet, handler)
	http.HandleFunc(relativePath, handler)
}

func (h *Http) POST(relativePath string, handler http.HandlerFunc) {
	h.registerPath(relativePath)

	handler = h.makeHandler(http.MethodPost, handler)
	http.HandleFunc(relativePath, handler)
}

func (h *Http) Run(handler http.Handler) {
	LogRun(h.addr) // log
	log.Fatal(http.ListenAndServe(h.addr, handler))
}
func (h *Http) RunTLS(handler http.Handler) {
	LogRunTLS(h.addr) // log
	log.Fatal(http.ListenAndServeTLS(h.addr, h.certFile, h.keyFile, handler))
}

// pattern: URL路径前缀（必须以/结尾），例如 "/asset/"
// relativePath: html中引用的静态文件路径，该路径不要求真实存在，例如 "asset"
// root: 静态文件的根目录，该目录真实存在，相对路径/绝对路径都可以，例如 "assets"
func (h *Http) Static(pattern, relativePath, root string) {
	http.Handle(pattern, http.StripPrefix(relativePath, http.FileServer(http.Dir(root))))
}
