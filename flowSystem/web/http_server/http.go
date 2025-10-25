package http_server

import (
	"log"
	"net/http"
)

type Path struct {
	path   string
	method string
}

type Paths []Path

// false 表示路径不存在，不合法
func (paths Paths) contains(path string) bool {
	for _, p := range paths {
		if p.path == path {
			return true
		}
	}
	return false
}
func (paths *Paths) registerPath(path, method string) {
	if !paths.contains(path) && !paths.containsMethod(method) { // 避免重复注册
		*paths = append(*paths, Path{path, method})
	}
}

func (paths Paths) containsMethod(method string) bool {
	for _, p := range paths {
		if p.method == method {
			return true
		}
	}
	return false
}

type Http struct {
	addr     string
	certFile string
	keyFile  string
	paths    Paths // 记录所有注册的路径，用于检查请求路径是否合法
}

func NewHttp(addr string, opts ...string) *Http {
	var h = &Http{addr: addr}
	if len(opts) == 2 {
		h.certFile = opts[0]
		h.keyFile = opts[1]
	}
	return h
}

func (h *Http) handle(httpMethod, relativePath string, handler http.HandlerFunc) {
	h.paths.registerPath(relativePath, httpMethod)

	handler = ChainHandlers(handler, Method(httpMethod), ValidPath(relativePath)) // ----------------
	http.HandleFunc(relativePath, handler)
}

func (h *Http) GET(relativePath string, handler http.HandlerFunc) {
	h.handle(http.MethodGet, relativePath, handler)
}

func (h *Http) POST(relativePath string, handler http.HandlerFunc) {
	h.handle(http.MethodPost, relativePath, handler)
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
