import type { OpenAIToolDefinition, ToolEntry, ToolGroup } from '$lib/types';
import { ToolsService } from '$lib/services/tools.service';
import { mcpStore } from '$lib/stores/mcp.svelte';
import {
	BuiltInTool,
	GlobSearchType,
	HealthCheckStatus,
	JsonSchemaType,
	ToolCallType,
	ToolSource
} from '$lib/enums';
import { config } from '$lib/stores/settings.svelte';
import {
	buildBrowserInfoToolDefinition,
	buildGetDatetimeToolDefinition,
	buildReadMediaToolDefinition,
	DISABLED_TOOL_KEYS_LOCALSTORAGE_KEY,
	buildSandboxToolDefinition,
	HOME_TILDE,
	TOOL_GROUP_LABELS,
	TOOL_SERVER_LABELS
} from '$lib/constants';
import {
	BuiltInTool,
	GlobSearchType,
	HealthCheckStatus,
	JsonSchemaType,
	ToolCallType,
	ToolSource
} from '$lib/enums';
import { ToolsService } from '$lib/services/tools.service';
// direct imports between stores, not via the barrel, to avoid circular deps
import { mcpStore } from '$lib/stores/mcp/index.svelte';
import { modelsStore } from '$lib/stores/models/index.svelte';
import { settingsStore } from '$lib/stores/settings/index.svelte';
import type { OpenAIToolDefinition, ToolEntry, ToolGroup } from '$lib/types';
import { buildSandboxToolDefinition } from '$lib/utils';
import { SvelteMap, SvelteSet } from 'svelte/reactivity';

/** Stable selection identity for a tool, shared by the disabled set and the permission store */

class ToolsStore {
	private _disabledTools = $state(new SvelteSet<string>());
	private _error = $state<string | null>(null);
	private _loading = $state(false);
	private _serverHome = $state<string | null | undefined>(undefined);
	private _serverTools = $state<OpenAIToolDefinition[]>([]);
	private _toolsEndpointUnreachable = $state(false);
	private _serverHome = $state<string | null | undefined>(undefined);

		try {
			const stored = localStorage.getItem(DISABLED_TOOL_KEYS_LOCALSTORAGE_KEY);

			if (stored) {
				const parsed = JSON.parse(stored);

				if (Array.isArray(parsed)) {
					for (const key of parsed) {
						if (typeof key === 'string') this._disabledTools.add(key);
					}
				}
			}
		} catch (err) {
			console.error('[ToolsStore] Failed to load disabled tools from localStorage:', err);
		}

		this.fetchServerTools();
	}

	isGroupFullyEnabled(group: ToolGroup): boolean {
		return group.tools.length > 0 && group.tools.every((t) => this.isToolEnabled(t.key));
	}

	isToolEnabled(key: string): boolean {
		return !this._disabledTools.has(key);
	}

	/**
	 * Absolute home directory on the server, resolved once per session via
	 * file_glob_search's `base` field (the server expands `~`). Anchors the
	 * directory picker's search scope and the `~` abbreviation of cwd
	 * displays. Returns null when tools are unavailable.
	 */
	async resolveServerHome(): Promise<string | null> {
		if (this._serverHome !== undefined) return this._serverHome;

		try {
			const res = await ToolsService.executeToolRaw(BuiltInTool.SERVER_FILE_GLOB_SEARCH, {
				limit: 1,
				max_depth: 1,
				path: HOME_TILDE,
				type: GlobSearchType.DIR
			});

			this._serverHome = typeof res.base === 'string' ? res.base : null;
		} catch {
			// searches still work via a literal `~`, only `~` abbreviation degrades
			this._serverHome = null;
		}

		return this._serverHome;
	}

	setToolEnabled(key: string, enabled: boolean): void {
		if (enabled) {
			this._disabledTools.delete(key);
		} else {
			this._disabledTools.add(key);
		}
	}

	toggleGroup(group: ToolGroup): void {
		const allEnabled = group.tools.every((t) => this.isToolEnabled(t.key));
		const target = !allEnabled;

		for (const tool of group.tools) {
			if (target) this._disabledTools.delete(tool.key);
			else this._disabledTools.add(tool.key);
		}
		this.persistDisabledTools();
	}

	toggleTool(key: string): void {
		if (this._disabledTools.has(key)) {
			this._disabledTools.delete(key);
		} else {
			this._disabledTools.add(key);
		}

		this.persistDisabledTools();
	}

	/** First canonical entry matching a tool name, runtime tool calls resolve by name */
	private findEntryByName(toolName: string): ToolEntry | null {
		for (const entry of this.allTools) {
			if (entry.definition.function.name === toolName) return entry;
		}

		return null;
	}

	/** Get MCP tools from health check data, used when live connections aren't established yet */
	private getMcpToolsFromHealthChecks(): {
		serverId: string;
		serverName: string;
		tools: { name: string; description?: string }[];
	}[] {
		const result: ReturnType<ToolsStore['getMcpToolsFromHealthChecks']> = [];

		for (const server of mcpStore.getServers()) {
			if (!server.enabled) continue;

			const health = mcpStore.getHealthCheckState(server.id);

			if (health.status === HealthCheckStatus.SUCCESS && health.tools.length > 0) {
				result.push({
					serverId: server.id,
					serverName: mcpStore.getServerLabel(server),
					tools: health.tools
				});
			}
		}

		return result;
	}

	private groupLabel(entry: ToolEntry): string {
		switch (entry.source) {
			case ToolSource.MCP:
				return entry.serverName ?? '';
			case ToolSource.CUSTOM:
				return TOOL_GROUP_LABELS[ToolSource.CUSTOM];
			case ToolSource.BROWSER:
				return TOOL_GROUP_LABELS[ToolSource.BROWSER];
			default:
				return TOOL_GROUP_LABELS[ToolSource.SERVER];
		}
	}

	private hasServerTool(name: BuiltInTool): boolean {
		return this._serverTools.some((def) => def.function.name === name);
	}

	private inferTypeFromDefault(value: unknown): string | undefined {
		if (typeof value === 'string') return 'string';

		if (typeof value === 'boolean') return 'boolean';

		if (typeof value === 'number') return Number.isInteger(value) ? 'integer' : 'number';

		if (Array.isArray(value)) return 'array';

		if (value !== null && typeof value === 'object') return 'object';

		return undefined;
	}

	private mcpDefinition(
		name: string,
		description: string | undefined,
		schema?: Record<string, unknown>
	): OpenAIToolDefinition {
		return {
			function: {
				description,
				name,
				parameters: schema ?? { properties: {}, required: [], type: JsonSchemaType.OBJECT }
			},
			type: ToolCallType.FUNCTION
		};
	}

	/** Normalize MCP tools from live connections when available, fall back to health check data */
	private mcpEntries(): {
		serverId: string;
		serverName: string;
		definition: OpenAIToolDefinition;
	}[] {
		const out: { serverId: string; serverName: string; definition: OpenAIToolDefinition }[] = [];
		const connections = mcpStore.getConnections();

		if (connections.size > 0) {
			for (const [serverId, connection] of connections) {
				const serverName = mcpStore.getServerDisplayName(serverId);

				for (const tool of connection.tools) {
					const rawSchema = (tool.inputSchema as Record<string, unknown>) ?? {
						properties: {},
						required: [],
						type: JsonSchemaType.OBJECT
					};

					out.push({
						definition: {
							function: {
								description: tool.description,
								name: tool.name,
								parameters: this.normalizeJsonSchema(rawSchema)
							},
							type: ToolCallType.FUNCTION
						},
						serverId,
						serverName
					});
				}
			}
		} else {
			for (const { serverId, serverName, tools } of this.getMcpToolsFromHealthChecks()) {
				for (const tool of tools) {
					out.push({
						definition: this.mcpDefinition(tool.name, tool.description),
						serverId,
						serverName
					});
				}
			}
		}

		return out;
	}

	/**
	 * Recursively normalize a JSON Schema object: infers `type` from `default`
	 * for properties / items that omit it, and descends into nested `properties`
	 * and `items`. Returns a new object -- does not mutate the input.
	 */
	private normalizeJsonSchema(schema: Record<string, unknown>): Record<string, unknown> {
		if (!schema || typeof schema !== 'object') return schema;

		const normalized: Record<string, unknown> = { ...schema };

		if (normalized.properties && typeof normalized.properties === 'object') {
			const props = normalized.properties as Record<string, Record<string, unknown>>;
			const normalizedProps: Record<string, Record<string, unknown>> = {};

			for (const [key, prop] of Object.entries(props)) {
				if (!prop || typeof prop !== 'object') {
					normalizedProps[key] = prop;

					continue;
				}

				const normalizedProp: Record<string, unknown> = { ...prop };

				if (!normalizedProp.type && normalizedProp.default !== undefined) {
					const inferred = this.inferTypeFromDefault(normalizedProp.default);

					if (inferred) normalizedProp.type = inferred;
				}

				if (normalizedProp.properties) {
					Object.assign(
						normalizedProp,
						this.normalizeJsonSchema(normalizedProp as Record<string, unknown>)
					);
				}

				if (normalizedProp.items && typeof normalizedProp.items === 'object') {
					normalizedProp.items = this.normalizeJsonSchema(
						normalizedProp.items as Record<string, unknown>
					);
				}

				normalizedProps[key] = normalizedProp;
			}
			normalized.properties = normalizedProps;
		}

		return normalized;
	}

	private mcpDefinition(
		name: string,
		description: string | undefined,
		schema?: Record<string, unknown>
	): OpenAIToolDefinition {
		return {
			type: ToolCallType.FUNCTION,
			function: {
				name,
				description,
				parameters: schema ?? { type: JsonSchemaType.OBJECT, properties: {}, required: [] }
			}
		};
	}

	get builtinTools(): OpenAIToolDefinition[] {
		return this._builtinTools;
	}

	get serverHome(): string | null {
		return this._serverHome ?? null;
	}

	get mcpTools(): OpenAIToolDefinition[] {
		return this.mcpEntries().map((e) => e.definition);
	}

	get frontendTools(): OpenAIToolDefinition[] {
		return config().jsSandboxEnabled
			? [buildSandboxToolDefinition(!!config().symbolicMathEnabled)]
			: [];
	}

	get customTools(): OpenAIToolDefinition[] {
		const raw = config().customJson;
		if (!raw || typeof raw !== 'string') return [];

		try {
			localStorage.setItem(
				DISABLED_TOOL_KEYS_LOCALSTORAGE_KEY,
				JSON.stringify([...this._disabledTools])
			);
		} catch {
			// ignore storage errors
		}
	}

	/**
	 * `read_media` runs in the browser on top of the server's `read_file`, so it
	 * exists only when that tool is served and the active model can perceive the
	 * bytes. The server cannot make this call - it does not know which model the
	 * conversation uses.
	 */
	private readMediaTool(): OpenAIToolDefinition | null {
		if (!this.hasServerTool(BuiltInTool.SERVER_READ_FILE)) return null;

		const model = modelsStore.selectedModelName ?? modelsStore.models[0]?.model ?? '';

		if (!model) return null;

		const vision = modelsStore.props.modelSupportsVision(model);
		const audio = modelsStore.props.modelSupportsAudio(model);

		if (!vision && !audio) return null;

		return buildReadMediaToolDefinition(vision, audio);
	}

	private toolKey(source: ToolSource, name: string, serverId?: string): string {
		switch (source) {
			case ToolSource.MCP:
				return serverId ? `mcp-${serverId}:${name}` : `mcp:${name}`;
			case ToolSource.CUSTOM:
				return `custom:${name}`;
			case ToolSource.BROWSER:
				return `browser:${name}`;
			default:
				return `server:${name}`;
		}
	}

	/**
	 * Absolute home directory on the server, resolved once per session via
	 * file_glob_search's `base` field (the server expands `~`). Anchors the
	 * directory picker's search scope and the `~` abbreviation of cwd
	 * displays. Returns null when tools are unavailable.
	 */
	async resolveServerHome(): Promise<string | null> {
		if (this._serverHome !== undefined) return this._serverHome;
		try {
			const res = await ToolsService.executeToolRaw(BuiltInTool.FILE_GLOB_SEARCH, {
				path: HOME_TILDE,
				type: GlobSearchType.DIR,
				max_depth: 1,
				limit: 1
			});
			this._serverHome = typeof res.base === 'string' ? res.base : null;
		} catch {
			// searches still work via a literal `~`, only `~` abbreviation degrades
			this._serverHome = null;
		}
		return this._serverHome;
	}
}

export const toolsStore = new ToolsStore();
