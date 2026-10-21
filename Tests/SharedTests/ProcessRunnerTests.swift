import XCTest
import Darwin
@testable import Shared

final class ProcessRunnerTests: XCTestCase {
    func testRunnerCanBeReusedForSequentialJobs() async throws {
        let runner = ProcessRunner()

        for percent in [25, 75] {
            var received: [IPCMessage] = []
            let json = #"{"protocol_version":1,"type":"progress","stage":"test","percent":\#(percent),"message":"running"}"#
            for try await message in runner.run(
                pythonPath: "/bin/sh",
                scriptPath: "-c",
                arguments: ["printf '%s\\n' '\(json)'"]
            ) {
                received.append(message)
            }

            XCTAssertEqual(received.count, 1)
            guard case .progress(_, let value, _) = received[0] else {
                return XCTFail("Expected a progress message")
            }
            XCTAssertEqual(value, percent)
        }
    }

    func testCancellationTerminatesDescendantProcess() async throws {
        let runner = ProcessRunner()
        let pidFile = FileManager.default.temporaryDirectory
            .appendingPathComponent("reelpack-child-\(UUID().uuidString).pid")
        defer { try? FileManager.default.removeItem(at: pidFile) }

        let stream = runner.run(
            pythonPath: "/bin/sh",
            scriptPath: "-c",
            arguments: ["sleep 30 & echo $! > '\(pidFile.path)'; wait"]
        )
        let consumer = Task {
            do {
                for try await _ in stream {}
            } catch {}
        }

        let childPID = try await waitForPID(in: pidFile)
        runner.cancel()

        let finished = expectation(description: "process tree stopped")
        Task {
            await consumer.value
            finished.fulfill()
        }
        await fulfillment(of: [finished], timeout: 2)
        XCTAssertNotEqual(kill(childPID, 0), 0)
    }

    func testImmediateCancellationFinishesPromptly() async throws {
        let runner = ProcessRunner()
        let stream = runner.run(
            pythonPath: "/bin/sh",
            scriptPath: "-c",
            arguments: ["sleep 30"]
        )

        runner.cancel()
        let consumer = Task {
            do {
                for try await _ in stream {}
            } catch {}
        }

        let finished = expectation(description: "cancelled stream finished")
        Task {
            await consumer.value
            finished.fulfill()
        }
        await fulfillment(of: [finished], timeout: 2)
    }

    private func waitForPID(in file: URL) async throws -> pid_t {
        for _ in 0..<100 {
            if let value = try? String(contentsOf: file, encoding: .utf8),
               let pid = pid_t(value.trimmingCharacters(in: .whitespacesAndNewlines)) {
                return pid
            }
            try await Task.sleep(nanoseconds: 20_000_000)
        }
        throw XCTSkip("Child process did not start")
    }

    func testRunnerIncludesStderrTailOnFailure() async {
        let runner = ProcessRunner()

        do {
            for try await _ in runner.run(
                pythonPath: "/bin/sh",
                scriptPath: "-c",
                arguments: ["echo 'useful failure detail' >&2; exit 7"]
            ) {}
            XCTFail("Expected the process to fail")
        } catch {
            XCTAssertTrue(error.localizedDescription.contains("exit code: 7"))
            XCTAssertTrue(error.localizedDescription.contains("useful failure detail"))
        }
    }
}
