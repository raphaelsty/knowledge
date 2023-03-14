launch:
	echo ${OPENAI_API_KEY} > mysecret.txt
	docker build --secret id=OPENAI_API_KEY,src=mysecret.txt -t knowledge .
	docker run -d --add-host host.docker.internal:host-gateway --name run_knowledge -p 8080:8080 knowledge