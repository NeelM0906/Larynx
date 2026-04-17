"""FastAPI dependency-injection helpers.

The app holds long-lived objects (worker client, DB engine, redis, latent
cache root, data dir) on ``app.state``; these helpers surface them as
FastAPI dependencies so routes stay decoupled from how they were constructed.
"""

from __future__ import annotations

import pathlib
from collections.abc import AsyncIterator

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from larynx_gateway.db.session import get_session
from larynx_gateway.services.latent_cache import LatentCache
from larynx_gateway.services.voice_library import VoiceLibrary
from larynx_gateway.workers_client.voxcpm_client import VoxCPMClient


def get_voxcpm_client(request: Request) -> VoxCPMClient:
    client: VoxCPMClient | None = getattr(request.app.state, "voxcpm_client", None)
    if client is None:
        raise RuntimeError("voxcpm_client not initialised (lifespan did not run?)")
    return client


def get_latent_cache(request: Request) -> LatentCache:
    cache: LatentCache | None = getattr(request.app.state, "latent_cache", None)
    if cache is None:
        raise RuntimeError("latent_cache not initialised (lifespan did not run?)")
    return cache


def get_data_dir(request: Request) -> pathlib.Path:
    data_dir: pathlib.Path | None = getattr(request.app.state, "data_dir", None)
    if data_dir is None:
        raise RuntimeError("data_dir not initialised (lifespan did not run?)")
    return data_dir


def get_design_ttl_s(request: Request) -> int:
    ttl: int | None = getattr(request.app.state, "design_ttl_s", None)
    if ttl is None:
        raise RuntimeError("design_ttl_s not initialised (lifespan did not run?)")
    return ttl


async def get_db_session() -> AsyncIterator[AsyncSession]:
    async for session in get_session():
        yield session


async def get_voice_library(
    session: AsyncSession = Depends(get_db_session),
    voxcpm: VoxCPMClient = Depends(get_voxcpm_client),
    cache: LatentCache = Depends(get_latent_cache),
    data_dir: pathlib.Path = Depends(get_data_dir),
    design_ttl_s: int = Depends(get_design_ttl_s),
) -> VoiceLibrary:
    return VoiceLibrary(
        session=session,
        voxcpm=voxcpm,
        cache=cache,
        data_dir=data_dir,
        design_ttl_s=design_ttl_s,
    )
